use extendr_api::prelude::*;
use ndarray_rand::*;


// Norm of vector, make generic later !
fn norm(v: ArrayView1<f64>) -> f64 {
    (v.dot(&v)).sqrt()
}

// Orthogonal projection of a set of vectors(mat) onto v
fn orth_proj(mat: ArrayView2<f64>, v: ArrayView1<f64>) -> Array1<f64> {
    v.dot(&mat)
    /***
     * Assumming |v|^2 = 1
     * Assumption needed as gradient won't be calculated properly otherwise
     */
}

// Calculate numerical gradient
fn gradient(mat: ArrayView2<f64>, v: ArrayView1<f64>, eps: f64) -> Array1<f64> {
    let mut grad: Array1<f64> = Array::zeros(v.len());
    let mut v_h: Array1<f64> = v.to_owned();

    for i in 0..v.len() {
        v_h[i] += eps;

        grad[i] = 
        (orth_proj(mat, v_h.view()).var(1.0) - orth_proj(mat, v).var(1.0))
        / eps;

        v_h[i] -= eps;
    }

    grad
}

// Find principal direction by iteratively finding vector with largest "projected variance"
fn find_pd(mat: ArrayView2<f64>, eps: f64, tol: f64, max_iter: i32) -> Array1<f64> {
    // Number of features
    let nfeat: usize = mat.nrows();

    // Initialize random vector
    let mut cur_v: Array1<f64> = Array1::random(
        nfeat.into_shape(), 
        rand_distr::Uniform::new(-1.0, 1.0));
    cur_v /= norm(cur_v.view());

    // Allocate space for gradient and prior iteration vector
    let mut grad: Array1<f64>;
    let mut old_v: Array1<f64>;

    // Find principal direction
    for i in 0..max_iter {
        old_v = cur_v.clone();

        // Update current vector with gradient
        grad = gradient(mat.view(), cur_v.view(), eps);
        cur_v += &grad;
        cur_v /= norm(cur_v.view());

        // If tolerance is reached, stop early
        if norm((&cur_v - &old_v).view()) < tol {
            println!("stopped early: {i}");
            break;
        }
    }

    cur_v
}

// Remove orthogonal projection from matrix inplace
fn remove_op(mut mat: ArrayViewMut2<f64>, v: ArrayView1<f64>) {
    let o: Array1<f64> = orth_proj(mat.view(), v);

    // Iterating over rows is better when nfeats < nobs
    Zip::from(mat.rows_mut())
    .and(v)
    .for_each(|mut row, &vi| row -= &(vi * &o));
}

// @export
#[extendr]
fn my_pca(mat: ArrayView2<f64>, k: i32, eps: f64, tol: f64, max_iter: i32) -> Robj {
    let mut mat: Array2<f64> = mat.to_owned(); // Copy R Matrix
    //let nfeat: usize = mat.nrows();
    let nobs: usize = mat.ncols();

    // Checks
    if k > mat.nrows() as i32 {
        throw_r_error("
        Desired number of principal components(k) greater than number of features present
         in data matrix");
    }

    // Allocate vector for principal directions and components
    //let mut p_dirs: Vec<f64> = Vec::new();
    let mut p_comps: Vec<f64> = Vec::new();

    for _ in 0..k {
        // Get principal direction
        let pd: Array1<f64> = find_pd(mat.view(), eps, tol, max_iter);

        // Get principal component
        let o: Array1<f64> = orth_proj(mat.view(), pd.view());

        // Remove projection from matrix
        remove_op(mat.view_mut(), pd.view());

        // Store principal pairs
        //p_dirs.extend(pd.into_raw_vec());
        p_comps.extend(o.into_raw_vec());
    }

    //let p_dirs: Array2<f64> = Array2::from_shape_vec((nfeat, k as usize).f(), p_dirs).unwrap();
    let p_comps: Array2<f64> = Array2::from_shape_vec((nobs, k as usize).f(), p_comps).unwrap();

    p_comps.try_into().unwrap()
}



// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod mypca;
    fn my_pca;
}
