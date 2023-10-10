
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "dot_product.h"

using namespace std;
namespace py = pybind11;


py::array_t<float> dot_product_vectors(py::array_t<float, py::array::c_style | py::array::forcecast> vector_one, py::array_t<float, py::array::c_style | py::array::forcecast> vector_two)
{
  dot_product_cu_parameters parameters;
  auto pbuf1 = vector_one.request();
  auto pbuf2 = vector_two.request();
  if (pbuf1.ndim != 2 || pbuf2.ndim != 2 )
    throw std::runtime_error("inputs must have a shape of (m, n)");
	parameters.vec_one_row = pbuf1.shape[0];
	parameters.vec_one_col = pbuf1.shape[1];
	parameters.vec_two_row = pbuf2.shape[0];
	parameters.vec_two_col = pbuf2.shape[1];
	if (parameters.vec_one_col != parameters.vec_two_row) {
		printf("Vector One :: %d x %d\n", parameters.vec_one_row, parameters.vec_one_col);
		printf("Vector Two :: %d x %d\n", parameters.vec_two_row, parameters.vec_two_col);
    throw std::runtime_error("Invalid Vectors Size for Dot Product");
	}
  parameters.pbuf1_ptr = static_cast<float *>(pbuf1.ptr);
  parameters.pbuf2_ptr = static_cast<float *>(pbuf2.ptr);

  py::array_t<float> ret_val( {parameters.vec_one_row, parameters.vec_two_col} );
  auto pbuf_ret = ret_val.request();
  parameters.pbuf_ret_ptr = static_cast<float *>(pbuf_ret.ptr);
  dot_product_cu(parameters);
	return ret_val;
}

PYBIND11_MODULE(net_ext, m) {
	m.def("dot_product_vectors", &dot_product_vectors, "" );


}


