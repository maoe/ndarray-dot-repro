use cblas_sys::{cblas_dgemm, CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans};
use ndarray::array;

extern crate blas_src as _;
extern crate openblas_src as _;

fn main() {
    let lhs_trans = CblasNoTrans;
    let rhs_trans = CblasNoTrans;
    let m = 9;
    let n = 9;
    let k = 9;
    let alpha = 1.0;
    let lhs = array![
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ];
    let lhs_stride = 9;
    let rhs = array![
        [
            0.9992622351823005,
            0.0009331243412841866,
            -0.004920948935161985,
            0.009453311595934506,
            -0.04232877039009844,
            0.000052283438296009615,
            -0.0006845176208477951,
            0.000898206291128021,
            -0.004778240468201882
        ],
        [
            0.0018761188100995551,
            0.9976270864551335,
            0.012513858941641174,
            -0.024039552005432698,
            0.10764108077812706,
            -0.00013295557024485316,
            0.001740712424213594,
            -0.0022841177536626375,
            0.012150954621997426
        ],
        [
            -0.0032083816985441957,
            0.0040579585624288105,
            0.9785998435757943,
            0.041110434094170686,
            -0.184078786333225,
            0.00022736951199293088,
            -0.002976821006330104,
            0.003906107416396626,
            -0.020779547765948165
        ],
        [
            0.010239549654644506,
            -0.012950974073731226,
            0.06829859565096116,
            0.8687960564597852,
            0.5874874158149637,
            -0.0007256497595221631,
            0.009500523744147034,
            -0.012466341166550403,
            0.06631792322186275
        ],
        [
            -0.01473358075459916,
            0.018635020953239743,
            -0.09827413395985797,
            0.18878798020133603,
            0.15467051040722202,
            0.001044129838915893,
            -0.013670204112139425,
            0.017937687738878497,
            -0.09542416514611302
        ],
        [
            0.00010712124508406634,
            -0.0001354868636434887,
            0.0007145070682191212,
            -0.0013725925715485292,
            0.006146010867262541,
            0.9999924086011244,
            0.0000993899113485083,
            -0.0001304168671908251,
            0.0006937862256171745
        ],
        [
            -0.00008726089698083179,
            0.00011036751151811463,
            -0.0005820369957706817,
            0.0011181130212644143,
            -0.005006536478458479,
            0.000006183948615465473,
            0.9999190370424811,
            0.000106237495686037,
            -0.0005651578107852428
        ],
        [
            0.00047538337811343435,
            -0.0006012645099325517,
            0.0031708442476501334,
            -0.00609130049715229,
            0.02727480814574412,
            -0.00003368916071932693,
            0.0004410732135363926,
            0.9994212351542452,
            0.0030788891537213816
        ],
        [
            0.00026427065974992704,
            -0.0003342493154781206,
            0.001762705933506178,
            -0.00338621852431151,
            0.015162355006675371,
            -0.00001872816160516541,
            0.00024519727551658134,
            -0.0003217415136277619,
            1.0017115871219138
        ]
    ];
    let rhs_stride = 9;
    let beta = 0.0;
    let mut c = array![
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ];
    let c_stride = 9;
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
