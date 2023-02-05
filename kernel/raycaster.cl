const sampler_t sp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
const sampler_t sp2 = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
const sampler_t sp3 = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

#ifndef M_PI
# define M_PI 3.141592653589793115998
#endif

#ifndef M_PI_F
# define M_PI_F 3.14159274101257f
#endif

__inline float eval(float3 p_in, __read_only image3d_t vol);
__inline float3 eval_g(float3 p_in, __read_only image3d_t vol);

__inline float4 MDotV(float16 m, float4 v) 
{
	return (float4)(dot(m.s0123, v),dot(m.s4567, v), dot(m.s89ab, v), dot(m.scdef, v));
}

__kernel void genRay(__global float8* ray, float16 MVP, float fov)
{
	int2 idx = (int2)(get_global_id(0), get_global_id(1));
	int2 sz  = (int2)(get_global_size(0), get_global_size(1));

	int id = sz.x*idx.y+idx.x;

	float4 b;
	float4 e;

    if(fov == 0) {
        b = (float4)(convert_float2(idx*2)/convert_float2(sz-1)-1.0f, -1.0f, 1.0f); //[-1...1]^3
        e = (float4)(b.xy, 1.0f, 1.0f);
    } else {
        b = (float4)(0, 0, -1-sqrt(2.0f)/tan(fov*0.5f/180.0f*M_PI_F), 1.0f);
        e = (float4)(convert_float2(idx*2)/convert_float2(sz-1)-1.0f, -1.0f, 1.0f);
    }

	b = MDotV(MVP, b);
    e = MDotV(MVP, e);

    float4 d = normalize(e-b);

    float2 hit_yz = (float2)(-1-b.x, 1-b.x)/d.x;
    float2 hit_zx = (float2)(-1-b.y, 1-b.y)/d.y;
    float2 hit_xy = (float2)(-1-b.z, 1-b.z)/d.z;

    hit_yz = (float2)(min(hit_yz.x, hit_yz.y), max(hit_yz.x, hit_yz.y));
    hit_zx = (float2)(min(hit_zx.x, hit_zx.y), max(hit_zx.x, hit_zx.y));
    hit_xy = (float2)(min(hit_xy.x, hit_xy.y), max(hit_xy.x, hit_xy.y));

    float2 bound =  (float2)(
        max(max(hit_yz.x, hit_zx.x), hit_xy.x),
        min(min(hit_yz.y, hit_zx.y), hit_xy.y));
   
    if(any(isnan(bound)) || any(isinf(bound)) || (bound.x > bound.y))
        bound = (float2)(0);

    ray[id] = (float8)(b.xyz+d.xyz*bound.x, 1, b.xyz+d.xyz*bound.y, 1);
}

__kernel void raycast(__write_only image2d_t Position, __read_only image3d_t vol, __global float8* Rays, float4 scale, int4 dim, float level)
{
    int2 id = (int2)(get_global_id(0), get_global_id(1));
    int2 sz  = (int2)(get_global_size(0), get_global_size(1));

	float8 ray = Rays[sz.x*id.y+id.x];

	float3 fdim = convert_float3(dim.xyz-1);
	float3 p = fdim*(0.5f*ray.s012+0.5f); // [0...fdim]^3
    float3 e = fdim*(0.5f*ray.s456+0.5f); // [0...fdim]^3
	float3 p_prev;

	float step = 0.5f;
    float max_ray_len = distance(p,e);
    float voxel = eval(p, vol);
    float voxel_prev = voxel;
    
    float orientation = 2.f*convert_float(voxel < level)-1.f;     // equivalent to (voxel<level?1:-1)
    float3 dir = normalize(e-p)*step;

    int max_iter = min(100000, convert_int(max_ray_len/step)+1);//convert_int(ray.w*fdim.z/dir.z));
    float4 val = (float4)(0);

    int i=0;

    float3 inverse_scale = 1.f/(fdim.xyz-1.f);
    for(i=0; i<max_iter; i++) {
    	p = p+dir.xyz;
    	voxel = eval(p, vol);
		if(orientation*voxel > orientation*level) {
            // One step of Regula Falsi
            if(fabs(voxel-voxel_prev) > 1E-4) 
                p = (p*(voxel_prev-level) - p_prev*(voxel-level))/(voxel_prev-voxel);
            val = (float4)((p*inverse_scale), orientation);
            break;
        }
        voxel_prev=voxel;
        p_prev=p;
    }
    
    write_imagef(Position, id, val);
}

__kernel void evalGradient(__write_only image2d_t Gradient, __read_only image3d_t vol, __read_only image2d_t Position, int4 dim)
{
    int2 id = (int2)(get_global_id(0), get_global_id(1));
    float4 p = read_imagef(Position, sp2, id)*(float4)(convert_float3(dim.xyz-1), 1);

    float3 grad = (float3)(0);
    #if 1
    if(p.w!=0)  grad = eval_g(p.xyz, vol);
    #else
    if(p.w!=0){
        float delta = 0.1;
        float d011 = eval(p.xyz-(float3)(delta, 0, 0), vol);
        float d211 = eval(p.xyz+(float3)(delta, 0, 0), vol);
        float d101 = eval(p.xyz-(float3)(0, delta, 0), vol);
        float d121 = eval(p.xyz+(float3)(0, delta, 0), vol);
        float d110 = eval(p.xyz-(float3)(0, 0, delta), vol);
        float d112 = eval(p.xyz+(float3)(0, 0, delta), vol);
        grad = (float3)(d211-d011, d121-d101, d112-d110)/(2*delta);
    }

    #endif

    write_imagef(Gradient, id, (float4)(grad,1));
}


__kernel void evalFiniteGradient(__write_only image2d_t Gradient, __read_only image3d_t vol, __read_only image2d_t Position, int4 dim)
{
    int2 id = (int2)(get_global_id(0), get_global_id(1));
    float4 p = read_imagef(Position, sp2, id)*(float4)(convert_float3(dim.xyz-1), 1);

    float3 grad = (float3)(0);
    if(p.w!=0){
        float delta = 0.1;
        float d011 = eval(p.xyz-(float3)(delta, 0, 0), vol);
        float d211 = eval(p.xyz+(float3)(delta, 0, 0), vol);
        float d101 = eval(p.xyz-(float3)(0, delta, 0), vol);
        float d121 = eval(p.xyz+(float3)(0, delta, 0), vol);
        float d110 = eval(p.xyz-(float3)(0, 0, delta), vol);
        float d112 = eval(p.xyz+(float3)(0, 0, delta), vol);
        grad = (float3)(d211-d011, d121-d101, d112-d110)/(2*delta);
    }

    write_imagef(Gradient, id, (float4)(grad,1));
}

__kernel void hessian(__global float8* dxx, __read_only image2d_t vol, __global float4* pos)
{
    
}
