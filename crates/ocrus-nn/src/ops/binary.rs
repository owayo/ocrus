use crate::tensor::NdTensor;

pub fn add(a: &NdTensor<f32>, b: &NdTensor<f32>) -> NdTensor<f32> {
    NdTensor::broadcast_binary_op(a, b, |x, y| x + y)
}

pub fn sub(a: &NdTensor<f32>, b: &NdTensor<f32>) -> NdTensor<f32> {
    NdTensor::broadcast_binary_op(a, b, |x, y| x - y)
}

pub fn mul(a: &NdTensor<f32>, b: &NdTensor<f32>) -> NdTensor<f32> {
    NdTensor::broadcast_binary_op(a, b, |x, y| x * y)
}

pub fn div(a: &NdTensor<f32>, b: &NdTensor<f32>) -> NdTensor<f32> {
    NdTensor::broadcast_binary_op(a, b, |x, y| x / y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_same_shape() {
        let a = NdTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = NdTensor::from_vec(vec![4.0, 5.0, 6.0], &[3]);
        let c = add(&a, &b);
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_broadcast_scalar() {
        let a = NdTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = NdTensor::from_vec(vec![10.0], &[1]);
        let c = add(&a, &b);
        assert_eq!(c.data, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_sub_broadcast() {
        let a = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = NdTensor::from_vec(vec![1.0, 0.0, -1.0], &[3]);
        let c = sub(&a, &b);
        assert_eq!(c.shape, vec![2, 3]);
        assert_eq!(c.data, vec![0.0, 2.0, 4.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_mul_div() {
        let a = NdTensor::from_vec(vec![2.0, 4.0, 6.0], &[3]);
        let b = NdTensor::from_vec(vec![2.0], &[1]);
        let c = mul(&a, &b);
        assert_eq!(c.data, vec![4.0, 8.0, 12.0]);
        let d = div(&a, &b);
        assert_eq!(d.data, vec![1.0, 2.0, 3.0]);
    }
}
