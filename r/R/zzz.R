.onLoad <- function(libname, pkgname) {
  reticulate::py_require(
    "lgtm @ https://github.com/yuanx749/lgtm/archive/refs/tags/v0.2.0.zip",
    python_version = "==3.10.14"
  )
}
