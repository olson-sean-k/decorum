fn prelude() {
    println!("cargo:rustc-check-cfg=cfg(nightly)");
}

#[rustversion::not(nightly)]
fn main() {
    self::prelude();
}

#[rustversion::nightly]
fn main() {
    self::prelude();
    println!("cargo:rustc-cfg=nightly");
}
