
struct dot_product_cu_parameters
{
	int vec_one_row;
	int vec_one_col;
	int vec_two_row;
	int vec_two_col;
  float *pbuf1_ptr;
  float *pbuf2_ptr;
  float *pbuf_ret_ptr;


};

void dot_product_cu(dot_product_cu_parameters parameters);

#define NET_DEBUG 0

