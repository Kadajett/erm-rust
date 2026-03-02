//! GPU↔CPU bridge for transferring between burn tensors and Vec<f32>.
//!
//! The colony logic (ant sampling, merge, pheromone updates) operates on
//! `Vec<f32>` on CPU. The burn scorer operates on `Tensor<B>` on GPU.
//! This module provides the conversion functions.
//!
//! # Functions
//!
//! - [`tensor_to_vec`]: GPU `Tensor<B,3>` → CPU `Vec<f32>`
//! - [`vec_to_tensor`]: CPU `Vec<f32>` → GPU `Tensor<B,3>`
//! - [`tokens_to_tensor`]: CPU `&[u32]` → GPU `Tensor<B,2,Int>`
//! - [`tensor2d_to_vec`]: GPU `Tensor<B,2>` → CPU `Vec<f32>`

use burn::prelude::*;

use erm_core::error::{ErmError, ErmResult};

/// Transfer a 3D burn tensor from GPU to CPU as a flat `Vec<f32>`.
///
/// # Arguments
///
/// - `tensor`: burn tensor with shape `[B, L, V]`.
///
/// # Returns
///
/// Flat `Vec<f32>` of length `B * L * V`, row-major.
///
/// # Errors
///
/// Returns [`ErmError::ShapeMismatch`] if the tensor data cannot be read as f32.
pub fn tensor_to_vec<B: Backend>(tensor: Tensor<B, 3>) -> ErmResult<Vec<f32>> {
    let data = tensor.into_data();
    data.as_slice::<f32>()
        .map(|s| s.to_vec())
        .map_err(|e| ErmError::ShapeMismatch {
            expected: "f32 tensor data".to_string(),
            got: format!("conversion error: {e}"),
        })
}

/// Transfer a 2D burn tensor from GPU to CPU as a flat `Vec<f32>`.
///
/// # Arguments
///
/// - `tensor`: burn tensor with shape `[B, L]`.
///
/// # Returns
///
/// Flat `Vec<f32>` of length `B * L`, row-major.
///
/// # Errors
///
/// Returns [`ErmError::ShapeMismatch`] if the tensor data cannot be read as f32.
pub fn tensor2d_to_vec<B: Backend>(tensor: Tensor<B, 2>) -> ErmResult<Vec<f32>> {
    let data = tensor.into_data();
    data.as_slice::<f32>()
        .map(|s| s.to_vec())
        .map_err(|e| ErmError::ShapeMismatch {
            expected: "f32 tensor data".to_string(),
            got: format!("conversion error: {e}"),
        })
}

/// Transfer a flat CPU `Vec<f32>` to a 3D burn tensor on the given device.
///
/// # Arguments
///
/// - `data`: flat `Vec<f32>` of length `shape[0] * shape[1] * shape[2]`.
/// - `shape`: `[B, L, V]` target shape.
/// - `device`: burn device to place the tensor on.
///
/// # Returns
///
/// `Tensor<B, 3>` with shape `[B, L, V]`.
///
/// # Errors
///
/// Returns [`ErmError::ShapeMismatch`] if `data.len() != B * L * V`.
pub fn vec_to_tensor<B: Backend>(
    data: &[f32],
    shape: [usize; 3],
    device: &B::Device,
) -> ErmResult<Tensor<B, 3>> {
    let expected = shape[0] * shape[1] * shape[2];
    if data.len() != expected {
        return Err(ErmError::ShapeMismatch {
            expected: format!("[{}, {}, {}] = {expected}", shape[0], shape[1], shape[2]),
            got: format!("{}", data.len()),
        });
    }
    let tensor_data = TensorData::new(data.to_vec(), shape);
    Ok(Tensor::<B, 3>::from_data(tensor_data, device))
}

/// Convert token ids (`&[u32]`) to a 2D burn Int tensor.
///
/// # Arguments
///
/// - `tokens`: flat token ids of length `batch_size * seq_len`.
/// - `batch_size`: batch dimension.
/// - `seq_len`: sequence dimension.
/// - `device`: burn device.
///
/// # Returns
///
/// `Tensor<B, 2, Int>` with shape `[batch_size, seq_len]`.
///
/// # Errors
///
/// Returns [`ErmError::ShapeMismatch`] if `tokens.len() != batch_size * seq_len`.
pub fn tokens_to_tensor<B: Backend>(
    tokens: &[u32],
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
) -> ErmResult<Tensor<B, 2, Int>> {
    let expected = batch_size * seq_len;
    if tokens.len() != expected {
        return Err(ErmError::ShapeMismatch {
            expected: format!("[{batch_size}, {seq_len}] = {expected}"),
            got: format!("{}", tokens.len()),
        });
    }
    let i64_data: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
    let tensor_data = TensorData::new(i64_data, [batch_size, seq_len]);
    Ok(Tensor::<B, 2, Int>::from_data(tensor_data, device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_tensor_to_vec_roundtrip() {
        let device = Default::default();
        let original = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [1, 2, 3];
        let tensor_data = TensorData::new(original.clone(), shape);
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);

        let result = tensor_to_vec(tensor).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn test_vec_to_tensor_roundtrip() {
        let device = Default::default();
        let original = vec![1.5_f32, -2.0, 3.14, 0.0, 100.0, -0.001];
        let shape = [2, 1, 3];

        let tensor = vec_to_tensor::<TestBackend>(&original, shape, &device).unwrap();
        let result = tensor_to_vec(tensor).unwrap();

        assert_eq!(result.len(), original.len());
        for (a, b) in result.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_vec_to_tensor_shape_mismatch() {
        let device = Default::default();
        let data = vec![1.0_f32; 5]; // doesn't match 2*3*4 = 24
        let result = vec_to_tensor::<TestBackend>(&data, [2, 3, 4], &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_tokens_to_tensor_basic() {
        let device = Default::default();
        let tokens = vec![0_u32, 5, 10, 15];
        let tensor = tokens_to_tensor::<TestBackend>(&tokens, 2, 2, &device).unwrap();
        assert_eq!(tensor.dims(), [2, 2]);

        let data = tensor.into_data();
        let vals = data.as_slice::<i64>().unwrap();
        assert_eq!(vals, &[0, 5, 10, 15]);
    }

    #[test]
    fn test_tokens_to_tensor_shape_mismatch() {
        let device = Default::default();
        let tokens = vec![0_u32; 5]; // doesn't match 2 * 3 = 6
        let result = tokens_to_tensor::<TestBackend>(&tokens, 2, 3, &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor2d_to_vec_roundtrip() {
        let device = Default::default();
        let original = vec![0.5_f32, 0.8, 0.3, 0.1];
        let tensor_data = TensorData::new(original.clone(), [2, 2]);
        let tensor = Tensor::<TestBackend, 2>::from_data(tensor_data, &device);

        let result = tensor2d_to_vec(tensor).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn test_large_tensor_roundtrip() {
        let device = Default::default();
        let b = 4;
        let l = 16;
        let v = 32;
        let original: Vec<f32> = (0..b * l * v).map(|i| i as f32 * 0.01).collect();

        let tensor = vec_to_tensor::<TestBackend>(&original, [b, l, v], &device).unwrap();
        let result = tensor_to_vec(tensor).unwrap();

        assert_eq!(result.len(), original.len());
        for (a, b) in result.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch at value: {a} vs {b}");
        }
    }
}
