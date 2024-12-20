fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=src/cuda_kernels/compatibility.cuh");
        println!("cargo:rerun-if-changed=src/cuda_kernels/cuda_utils.cuh");
        println!("cargo:rerun-if-changed=src/cuda_kernels/binary_op_macros.cuh");

        let builder = bindgen_cuda::Builder::default();
        println!("cargo:info={builder:?}");
        let bindings = builder.build_ptx().unwrap();
        bindings.write("src/cuda_kernels/mod.rs").unwrap();
    }
}
