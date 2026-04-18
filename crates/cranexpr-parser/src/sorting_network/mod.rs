//! Sorting network generation.
//!
//! Provides comparator pair sequences for sorting N elements via two strategies:
//!
//! - **N ≤ 64**: Precomputed optimal networks from
//!   [Bert Dobbelaere's tables](https://bertdobbelaere.github.io/sorting_networks_extended.html).
//! - **N > 64**: Recursive [Batcher odd-even mergesort](https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort).

mod optimal;

/// Returns the comparator pairs for a sorting network of size `n`.
///
/// For `n <= 64`, uses precomputed optimal networks. For `n > 64`, falls back
/// to Batcher's odd-even mergesort.
///
/// Returns `None` if `n < 2`.
#[must_use]
pub fn sorting_network(n: usize) -> Option<Vec<(usize, usize)>> {
  if n < 2 {
    return None;
  }

  if let Some(view) = optimal::lookup(n) {
    return Some(
      view
        .pairs()
        .iter()
        .map(|&(a, b)| (a as usize, b as usize))
        .collect(),
    );
  }

  Some(odd_even_mergesort(n))
}

/// Generates a Batcher odd-even mergesort network for `n` elements.
fn odd_even_mergesort(n: usize) -> Vec<(usize, usize)> {
  let p = n.next_power_of_two();
  let mut pairs = Vec::new();
  oem_sort(&mut pairs, 0, p);

  // Remove pairs referencing indices beyond n.
  pairs.retain(|&(a, b)| a < n && b < n);
  pairs
}

/// Recursively generates an odd-even mergesort.
fn oem_sort(pairs: &mut Vec<(usize, usize)>, lo: usize, n: usize) {
  if n > 1 {
    let m = n / 2;
    oem_sort(pairs, lo, m);
    oem_sort(pairs, lo + m, m);
    oem_merge(pairs, lo, n, 1);
  }
}

/// Recursively generates an odd-even mergesort for `n` elements starting at
/// `lo` with step size `step`.
fn oem_merge(pairs: &mut Vec<(usize, usize)>, lo: usize, n: usize, step: usize) {
  if n == 2 {
    pairs.push((lo, lo + step));
  } else if n > 2 {
    let double_step = step * 2;

    // Merge odd and even subsequences.
    oem_merge(pairs, lo, n / 2, double_step);
    oem_merge(pairs, lo + step, n / 2, double_step);

    // Compare-and-swap adjacent pairs.
    for i in (lo + step..lo + (n - 1) * step).step_by(double_step) {
      pairs.push((i, i + step));
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn verify_network(n: usize, pairs: &[(usize, usize)]) {
    assert!(n <= 20, "verification too expensive for n={n}");
    for mask in 0..(1u64 << n) {
      let mut vals: Vec<u8> = (0..n).map(|i| ((mask >> i) & 1) as u8).collect();
      for &(a, b) in pairs {
        if vals[a] > vals[b] {
          vals.swap(a, b);
        }
      }
      for w in vals.windows(2) {
        assert!(w[0] <= w[1], "failed for n={n}, mask={mask:#b}");
      }
    }
  }

  #[test]
  fn test_optimal_networks_valid() {
    for n in 2..=16 {
      let pairs = sorting_network(n).unwrap();
      verify_network(n, &pairs);
    }
  }

  #[test]
  fn test_oem_fallback() {
    for n in [65, 70, 100, 128] {
      let pairs = sorting_network(n).unwrap();
      for pattern in [
        (0..n).collect::<Vec<_>>(),
        (0..n).rev().collect::<Vec<_>>(),
        (0..n).map(|i| if i % 2 == 0 { n - i } else { i }).collect(),
      ] {
        let mut vals = pattern;
        for &(a, b) in &pairs {
          if vals[a] > vals[b] {
            vals.swap(a, b);
          }
        }
        for w in vals.windows(2) {
          assert!(w[0] <= w[1], "OEM failed for n={n}");
        }
      }
    }
  }

  #[test]
  fn test_n_less_than_2_returns_none() {
    assert!(sorting_network(0).is_none());
    assert!(sorting_network(1).is_none());
  }
}
