use cblas_sys::{cblas_dgemm, CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans};

extern crate openblas_src as _;

fn main() {
    let lhs_trans = CblasNoTrans;
    let rhs_trans = CblasNoTrans;
    let m = 2;
    let n = 2;
    let k = 2;
    let alpha = 1.0;
    let lhs = [1.0, 0.0, 0.0, 1.0];
    let lhs_stride = 2;
    let rhs = [
        0.9992622351823005,
        0.0009331243412841866,
        0.0018761188100995551,
        0.9976270864551335,
    ];
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
