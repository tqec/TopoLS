//! CPython's `random` module, reproduced exactly: MT19937 with
//! `init_by_array` seeding from an integer, `getrandbits`, `_randbelow`
//! (rejection sampling) and `shuffle` (Fisher–Yates from the end).
//!
//! The Python driver seeds with `random.seed(int)`, shuffles a few lists per
//! seed and hands the state to the search; the Rust port must draw the very
//! same numbers to make the same decisions.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER: u32 = 0x8000_0000;
const LOWER: u32 = 0x7fff_ffff;

#[derive(Clone)]
pub struct PyRandom {
    mt: [u32; N],
    index: usize,
}

impl PyRandom {
    fn init_genrand(s: u32) -> PyRandom {
        let mut mt = [0u32; N];
        mt[0] = s;
        for i in 1..N {
            mt[i] = 1_812_433_253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        PyRandom { mt, index: N }
    }

    fn init_by_array(key: &[u32]) -> PyRandom {
        let mut r = PyRandom::init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = N.max(key.len());
        while k > 0 {
            r.mt[i] = (r.mt[i] ^ ((r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_mul(1_664_525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                r.mt[0] = r.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            r.mt[i] = (r.mt[i] ^ ((r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941))).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                r.mt[0] = r.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        r.mt[0] = 0x8000_0000;
        r.index = N;
        r
    }

    /// `random.seed(n)` for a non-negative integer `n`.
    pub fn seed(n: u64) -> PyRandom {
        // key = 32-bit words of |n|, little-endian; at least one word
        let mut key = Vec::new();
        let mut v = n;
        loop {
            key.push((v & 0xffff_ffff) as u32);
            v >>= 32;
            if v == 0 {
                break;
            }
        }
        PyRandom::init_by_array(&key)
    }

    fn genrand_u32(&mut self) -> u32 {
        if self.index >= N {
            let mt = &mut self.mt;
            for kk in 0..N - M {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
            }
            for kk in N - M..N - 1 {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + M - N] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
            }
            let y = (mt[N - 1] & UPPER) | (mt[0] & LOWER);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random.getrandbits(k)` for 0 < k <= 64.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!(k > 0 && k <= 64);
        if k <= 32 {
            return (self.genrand_u32() >> (32 - k)) as u64;
        }
        // words are filled little-endian; the last word is shifted
        let lo = self.genrand_u32() as u64;
        let k2 = k - 32;
        let hi = (self.genrand_u32() >> (32 - k2)) as u64;
        lo | (hi << 32)
    }

    /// `random._randbelow(n)`: uniform in [0, n) by rejection sampling.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = 64 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    /// `random.shuffle(x)`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i as u64 + 1) as usize;
            x.swap(i, j);
        }
    }

    /// Rebuild from `random.getstate()`'s 624 words and index.
    pub fn from_state(words: &[u32], index: usize) -> PyRandom {
        assert_eq!(words.len(), N);
        let mut mt = [0u32; N];
        mt.copy_from_slice(words);
        PyRandom { mt, index }
    }

    /// The first words of the MT state and the index (for tests against
    /// `random.getstate()`).
    pub fn state_words(&self) -> (&[u32; N], usize) {
        (&self.mt, self.index)
    }
}
