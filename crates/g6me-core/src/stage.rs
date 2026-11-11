//! Bounded stage channels: the hand-off between decode, DSP and encode.
//!
//! ## Why this exists
//!
//! A render has three workers with very different speeds: a decoder process, one
//! DSP thread, and an encoder process. Read-decode-write on a single thread means
//! the encoder sits idle while the DSP works and the decoder blocks on a full
//! pipe; the file renders at roughly the *sum* of the three costs. Handing blocks
//! through a small bounded queue instead makes the cost the *maximum* of the three,
//! which is what lets a 7.1 Atmos file play to a device in real time at all.
//!
//! Three properties are load-bearing:
//!
//! * **Ordered.** The downmixer is a cascade of recursive all-pass sections whose
//!   state carries from block to block, so the queue is strictly FIFO. Blocks are
//!   never split, duplicated or reordered — the DSP sees the same contiguous
//!   sample sequence it would see with no threads at all. This is the whole reason
//!   parallelism here cannot change the output.
//!
//! * **Bounded.** Capacity is counted in blocks, so peak memory is
//!   `queue_blocks * block * channels * 8` regardless of file length. Unbounded
//!   would let the decoder win a race against a device-paced monitor and park a
//!   whole album in RAM.
//!
//! * **Closeable from any side.** When one stage fails or the user cancels, it
//!   [`Tx::close`]s (or [`Rx::close`]s) and every peer unblocks on its next
//!   `send`/`recv` instead of waiting forever on a pipe nobody is draining.
//!   [`Tx`] also closes on drop, so a panicked stage cannot strand its consumer.
//!
//! Blocking is condvar-based, not polled: a stage that is starved sleeps, so four
//! concurrent files do not burn four cores spinning on an empty queue.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Create a stage channel holding at most `capacity` items.
pub fn stage<T: Send>(capacity: usize) -> (Tx<T>, Rx<T>) {
    let p = Arc::new(Pipe {
        q: Mutex::new(VecDeque::with_capacity(capacity.max(1))),
        full: Condvar::new(),
        empty: Condvar::new(),
        capacity: capacity.max(1),
        closed: AtomicBool::new(false),
    });
    (Tx { p: p.clone() }, Rx { p })
}

struct Pipe<T> {
    q: Mutex<VecDeque<T>>,
    full: Condvar,
    empty: Condvar,
    capacity: usize,
    closed: AtomicBool,
}

impl<T> Pipe<T> {
    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.empty.notify_all();
        self.full.notify_all();
    }
}

/// Producer end of a [`stage`].
pub struct Tx<T> {
    p: Arc<Pipe<T>>,
}

/// Consumer end of a [`stage`].
pub struct Rx<T> {
    p: Arc<Pipe<T>>,
}

impl<T> Tx<T> {
    /// Hand a block downstream, waiting while the queue is full.
    ///
    /// Returns `false` when the channel is closed — the consumer is gone, so the
    /// producer should stop and clean up rather than queue work for nobody.
    pub fn send(&self, item: T) -> bool {
        let mut q = self.p.q.lock().unwrap_or_else(|e| e.into_inner());
        while q.len() >= self.p.capacity && !self.p.closed.load(Ordering::Acquire) {
            let (next, _) = self
                .p
                .full
                .wait_timeout(q, std::time::Duration::from_millis(500))
                .unwrap_or_else(|e| e.into_inner());
            q = next;
        }
        if self.p.closed.load(Ordering::Acquire) {
            return false;
        }
        q.push_back(item);
        drop(q);
        self.p.empty.notify_one();
        true
    }

    /// End the stream: the consumer drains what is queued and then sees `None`.
    pub fn close(&self) {
        self.p.close();
    }

    pub fn is_closed(&self) -> bool {
        self.p.closed.load(Ordering::Acquire)
    }

    /// Items currently queued (also readable from the producer side, for a
    /// back-pressure meter in the GUI).
    pub fn backlog(&self) -> usize {
        self.p.q.lock().map(|q| q.len()).unwrap_or(0)
    }
}

impl<T> Drop for Tx<T> {
    fn drop(&mut self) {
        // Covers the early-return and the panic path: nobody can be left waiting
        // on a producer that no longer exists.
        self.p.close();
    }
}

impl<T> Drop for Rx<T> {
    fn drop(&mut self) {
        // A dropped consumer must be as loud as a closed one. Without this a
        // stage that *returns early* (its encoder child died, say) leaves its
        // producer parked in `send` forever with nobody left to read: the
        // 24-bit FLAC route hung exactly here when `flac` refused its options.
        self.p.close();
    }
}

impl<T> Rx<T> {
    /// Take the next block, waiting while the queue is empty.
    ///
    /// `None` means end of stream: the producer closed and the queue is drained.
    pub fn recv(&self) -> Option<T> {
        let mut q = self.p.q.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(v) = q.pop_front() {
                drop(q);
                self.p.full.notify_one();
                return Some(v);
            }
            if self.p.closed.load(Ordering::Acquire) {
                return None;
            }
            let (next, _) = self
                .p
                .empty
                .wait_timeout(q, std::time::Duration::from_millis(500))
                .unwrap_or_else(|e| e.into_inner());
            q = next;
        }
    }

    /// Abort: producers stop at their next `send`, and any already-queued items
    /// are discarded on the next [`Rx::drain_discard`].
    pub fn close(&self) {
        self.p.close();
    }

    pub fn is_closed(&self) -> bool {
        self.p.closed.load(Ordering::Acquire)
    }

    pub fn backlog(&self) -> usize {
        self.p.q.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Drop everything still queued (used when abandoning a render, so the
    /// encoder cannot be handed samples after the DSP gave up).
    pub fn drain_discard(&self) {
        if let Ok(mut q) = self.p.q.lock() {
            q.clear();
        }
        self.p.full.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    #[test]
    fn order_is_preserved_end_to_end() {
        // The DSP's state depends on this absolutely: a reordered block is a
        // different signal, not a slower one.
        let (tx, rx) = stage(4);
        let producer = thread::spawn(move || {
            for i in 0..1000u64 {
                if !tx.send(i) {
                    return false;
                }
            }
            true
        });
        let mut got = Vec::new();
        while let Some(v) = rx.recv() {
            got.push(v);
        }
        assert!(producer.join().unwrap());
        assert_eq!(got.len(), 1000);
        assert_eq!(got[..10], (0..10).collect::<Vec<_>>()[..]);
        assert_eq!(got[999], 999);
        assert_eq!((0..1000).zip(&got).filter(|(a, b)| *a != **b).count(), 0);
    }

    #[test]
    fn a_full_queue_blocks_the_producer() {
        let (tx, rx) = stage(2);
        let sent = Arc::new(AtomicUsize::new(0));
        let counter = sent.clone();
        let producer = thread::spawn(move || {
            for i in 0..50 {
                if !tx.send(i) {
                    break;
                }
                counter.fetch_add(1, Ordering::Relaxed);
            }
        });
        // Let it fill the queue and park; without blocking it would be done already.
        thread::sleep(std::time::Duration::from_millis(40));
        assert_eq!(sent.load(Ordering::Relaxed), 2, "capacity must be honoured");
        assert_eq!(rx.backlog(), 2);
        // Draining must wake it.
        let mut total = 0;
        while rx.recv().is_some() {
            total += 1;
            if total == 50 {
                break;
            }
        }
        producer.join().unwrap();
        assert_eq!(total, 50);
        assert_eq!(sent.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn closing_the_reader_unblocks_a_waiting_writer() {
        let (tx, rx) = stage::<u32>(1);
        assert!(tx.send(1));
        let t = {
            let tx_ref = Tx { p: tx.p.clone() };
            thread::spawn(move || tx_ref.send(2))
        };
        thread::sleep(std::time::Duration::from_millis(40));
        rx.close();
        assert!(!t.join().unwrap(), "a closed channel refuses work");
        assert!(tx.is_closed());
    }

    #[test]
    fn dropping_the_writer_ends_the_reader() {
        let (tx, rx) = stage::<u32>(4);
        drop(tx);
        assert_eq!(rx.recv(), None);
        assert_eq!(rx.recv(), None, "EOF is sticky and idempotent");
    }

    #[test]
    fn queued_items_survive_a_normal_close() {
        let (tx, rx) = stage(8);
        tx.send(7);
        tx.send(8);
        tx.close();
        assert_eq!(rx.recv(), Some(7));
        assert_eq!(rx.recv(), Some(8));
        assert_eq!(rx.recv(), None);
    }

    #[test]
    fn drain_discard_empties_without_wedging_the_writer() {
        let (tx, rx) = stage(2);
        tx.send(1);
        tx.send(2);
        rx.drain_discard();
        assert_eq!(rx.backlog(), 0);
        rx.close();
        // The parked producer must return, not hang.
        let t = {
            let tx_ref = Tx { p: tx.p.clone() };
            thread::spawn(move || tx_ref.send(3))
        };
        assert!(!t.join().unwrap());
    }

    /// The bug that hung a render: a consumer that *returns early* (its encoder
    /// child died) must wake the producer, not leave it parked forever.
    #[test]
    fn dropping_the_consumer_wakes_a_parked_producer() {
        let (tx, rx) = stage::<u32>(1);
        assert!(tx.send(1));
        let parked = {
            let tx2 = Tx { p: tx.p.clone() };
            std::thread::spawn(move || {
                // Second send must block (queue full), then be released by the drop.
                let a = tx2.send(2);
                let b = tx2.send(3);
                (a, b)
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(60));
        drop(rx);
        let (a, b) = parked
            .join()
            .expect("a producer must never be stranded by a vanished consumer");
        assert!(!a && !b, "a vanished consumer refuses work");
    }

    /// Same shape as the pipeline, where the encoder thread is the consumer.
    #[test]
    fn a_consumer_thread_that_errors_out_does_not_strand_the_producer() {
        let (tx, rx) = stage::<Vec<u8>>(2);
        let consumer = std::thread::spawn(move || {
            // Simulate "encoder died": consume one block, then bail with rx dropped.
            let first = rx.recv();
            drop(first);
            Err::<(), _>("encoder died")
        });
        let producer = std::thread::spawn(move || {
            let mut sent = 0;
            for i in 0..1000usize {
                if !tx.send(vec![i as u8; 4096]) {
                    break;
                }
                sent += 1;
            }
            sent
        });
        let _ = consumer.join();
        let sent = producer
            .join()
            .expect("producer must finish, not block on a dead consumer");
        assert!(sent < 1000, "it stopped early: {sent}");
    }

    /// The shape the render uses: two threads, bounded, no data loss.
    #[test]
    fn pipeline_stages_never_lose_or_reorder_a_block() {
        let (tx, rx) = stage(3);
        let n = 5000usize;
        let producer = thread::spawn(move || {
            for i in 0..n {
                assert!(tx.send(vec![i as u8; 16]));
            }
        });
        let consumer = thread::spawn(move || {
            let mut sum = 0usize;
            let mut count = 0;
            while let Some(b) = rx.recv() {
                assert_eq!(b.len(), 16);
                sum += b[0] as usize;
                count += 1;
            }
            (sum, count)
        });
        producer.join().unwrap();
        let (sum, count) = consumer.join().unwrap();
        assert_eq!(count, n);
        assert_eq!(sum, (0..n).map(|i| i % 256).sum::<usize>());
    }
}
