const sampler_t sp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;


__kernel void raycaster(__global int* buf, __global float2* lastOrbits, float4 bound, int max_iter)
{
	int2 idx = (int2)(get_global_id(0), get_global_id(1));
	int2 sz  = (int2)(get_global_size(0), get_global_size(1));

	int buff_pos = idx.x+(idx.y*sz.x);

	float2 p = convert_float2(idx)/convert_float2(sz-1)*(bound.yw - bound.xz)+bound.xz;
	float2 n = lastOrbits[buff_pos];
	int val = buf[buff_pos];
	int i = 0;
	for( ; i<max_iter; i++){
		if((n.x*n.x+n.y*n.y)>=4) break;
		n = (float2)(n.x*n.x-n.y*n.y, 2*n.x*n.y)+p;
	}

	lastOrbits[buff_pos] = n;
	buf[buff_pos] = val+i;
}

__kernel void shading(__write_only image2d_t frame_buff)
{
	int2 idx = (int2)(get_global_id(0), get_global_id(1));
	int2 sz = (int2)(get_global_size(0), get_global_size(1));
	
	write_imagef(frame_buff, idx, (float4)(1,0,1,0));
}
