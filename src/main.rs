use cblas_sys::{cblas_dgemm, CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans};

#[link(name = "openblas")]
extern "C" {}

fn main() {
    let lhs_trans = CblasNoTrans;
    let rhs_trans = CblasNoTrans;
    let m = 2;
    let n = 2;
    let k = 2;
    let alpha = 1.0;
    let lhs = [1.0, 0.0, 0.0, 1.0];
    let lhs_stride = 2;
    let rhs = [1.0, 1.0, 1.0, 1.0];
    let rhs_stride = 2;
    let beta = 0.0;
    let mut c = [0.0, 0.0, 0.0, 0.0];
    let c_stride = 2;
    unsafe {
        cblas_dgemm(
            CblasRowMajor,
            lhs_trans,
            rhs_trans,
            m,
            n,
            k,
            alpha,
            lhs.as_ptr() as *const _,
            lhs_stride,
            rhs.as_ptr() as *const _,
            rhs_stride,
            beta,
            c.as_mut_ptr() as *mut _,
            c_stride,
        );
    }
    println!("{:#?}", c);
}
