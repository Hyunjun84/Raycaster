#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

const sampler_t sp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void applyQuasiInterpolator_CC(__write_only image3d_t vol, __read_only image3d_t org_vol, float4 coef, int4 dim)
{
    int4 id = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 1);
    int4 lid = (int4)(get_local_id(0), get_local_id(1), get_local_id(2), 1);
    
    float val = 0.0f;

    if(any(id.xyz>=dim.xyz)) return;

    // 0th
    val = read_imagef(org_vol, sp, id).x*coef.s0;

    // First neighbor
    val += read_imagef(org_vol, sp, id + (int4)(1,0,0,0)).x*coef.s1;
    val += read_imagef(org_vol, sp, id + (int4)(0,1,0,0)).x*coef.s1;
    val += read_imagef(org_vol, sp, id + (int4)(0,0,1,0)).x*coef.s1;
    val += read_imagef(org_vol, sp, id + (int4)(-1,0,0,0)).x*coef.s1;
    val += read_imagef(org_vol, sp, id + (int4)(0,-1,0,0)).x*coef.s1;
    val += read_imagef(org_vol, sp, id + (int4)(0,0,-1,0)).x*coef.s1;

    // Second neighbor
    val += read_imagef(org_vol, sp, id + (int4)(0,1,1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(1,0,1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(1,1,0,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)( 0,-1,1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(-1, 0,1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(-1, 1,0,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(0, 1,-1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(1, 0,-1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(1,-1, 0,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)( 0,-1,-1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(-1, 0,-1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(-1,-1, 0,0)).x*coef.s2;

    write_imagef(vol, id, val);

}

#if 0 
__kernel void applyQuasiInterpolator_CC_loc(__write_only image3d_t vol, __read_only image3d_t org_vol, float4 coef, int4 dim)
{
    int4 id = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 1);
    int4 lid = (int4)(get_local_id(0), get_local_id(1), get_local_id(2), 1);
    
    float val = 0.0f;

    // 4^3 grid kernel reads 6^3 data
    __local float s[216];
    int lidx = 16*(lid.z+1) + 4*(lid.y+1) + (lid.x+1);
    s[lidx] = read_imagef(org_vol, sp, id).x;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(any(id.xyz>=dim.xyz)) return;

    write_imagef(vol, id, val);

}
#endif

__kernel void applyQuasiInterpolator_FCC(__write_only image3d_t vol, __read_only image3d_t org_vol, float4 coef, int4 dim)
{
    int4 id = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 1);
    int4 lid = (int4)(get_local_id(0), get_local_id(1), get_local_id(2), 1);
    
    if(any(id.xyz>=dim.xyz)) return;

    // (0, 0, 0) += (0, 0, 0)*c0 : ret.s0 + (0, 0, 0).s0*coef.s0
    // (0, 1, 1) += (0, 1, 1)*c0 : ret.s1 + (0, 0, 0).s1*coef.s0
    // (1, 0, 1) += (1, 0, 1)*c0 : ret.s2 + (0, 0, 0).s2*coef.s0
    // (1, 1, 0) += (1, 1, 0)*c0 : ret.s3 + (0, 0, 0).s3*coef.s0

    // (0, 0, 0) += (0, 1, 1)*c1 : ret.s0 + (0, 0, 0).s1*coef.s1
    // (0, 1, 1) += (1, 0, 1)*c1 : ret.s1 + (0, 0, 0).s2*coef.s1
    // (1, 0, 1) += (1, 1, 0)*c1 : ret.s2 + (0, 0, 0).s3*coef.s1
    // (1, 1, 0) += (0, 0, 0)*c1 : ret.s3 + (0, 0, 0).s0*coef.s1

    // (0, 0, 0) += (1, 0, 1)*c1 : ret.s0 + (0, 0, 0).s2*coef.s1
    // (0, 1, 1) += (1, 1, 0)*c1 : ret.s1 + (0, 0, 0).s3*coef.s1
    // (1, 0, 1) += (0, 0, 0)*c1 : ret.s2 + (0, 0, 0).s0*coef.s1
    // (1, 1, 0) += (0, 1, 1)*c1 : ret.s3 + (0, 0, 0).s1*coef.s1

    // (0, 0, 0) += (1, 1, 0)*c1 : ret.s0 + (0, 0, 0).s3*coef.s1    
    // (0, 1, 1) += (0, 0, 0)*c1 : ret.s1 + (0, 0, 0).s0*coef.s1
    // (1, 0, 1) += (0, 1, 1)*c1 : ret.s2 + (0, 0, 0).s1*coef.s1
    // (1, 1, 0) += (1, 0, 1)*c1 : ret.s3 + (0, 0, 0).s2*coef.s1
    float4 tmp = read_imagef(org_vol, sp, id);
    float4 ret = tmp*coef.s0;
    ret += (tmp.s1230 + tmp.s2301 + tmp.s3012)*coef.s1;


    // (0, 0, 0) += (-1, 0, 1)*c1 : ret.s0 + (-1, 0, 0).s2*coef.s1
    // (0, 0, 0) += (-1, 1, 0)*c1 : ret.s0 + (-1, 0, 0).s3*coef.s1
    // (0, 1, 1) += (-1, 0, 1)*c1 : ret.s1 + (-1, 0, 0).s2*coef.s1
    // (0, 1, 1) += (-1, 1, 0)*c1 : ret.s1 + (-1, 0, 0).s3*coef.s1

    // (0, 0, 0) += (-2, 0, 0)*c2 : ret.s0 + (-1, 0, 0).s0*coef.s2
    // (0, 1, 1) += (-2, 1, 1)*c2 : ret.s1 + (-1, 0, 0).s1*coef.s2
    // (1, 0, 1) += (-1, 0, 1)*c2 : ret.s2 + (-1, 0, 0).s2*coef.s2
    // (1, 1, 0) += (-1, 1, 0)*c2 : ret.s3 + (-1, 0, 0).s3*coef.s2
    tmp = read_imagef(org_vol, sp, id+(int4)(-1,0,0,0));
    ret.s01 += (tmp.s2+tmp.s3)*coef.s1;
    ret += tmp*coef.s2;


    // (1, 0, 1) += (2, 0, 0)*c1 : ret.s2 + (1, 0, 0).s0*coef.s1
    // (1, 0, 1) += (2, 1, 1)*c1 : ret.s2 + (1, 0, 0).s1*coef.s1
    // (1, 1, 0) += (2, 0, 0)*c1 : ret.s3 + (1, 0, 0).s0*coef.s1
    // (1, 1, 0) += (2, 1, 1)*c1 : ret.s3 + (1, 0, 0).s1*coef.s1
    
    // (0, 0, 0) += (2, 0, 0)*c2 : ret.s0 + (1, 0, 0).s0*coef.s2
    // (0, 1, 1) += (2, 1, 1)*c2 : ret.s1 + (1, 0, 0).s1*coef.s2
    // (1, 0, 1) += (3, 0, 1)*c2 : ret.s2 + (1, 0, 0).s2*coef.s2
    // (1, 1, 0) += (3, 1, 0)*c2 : ret.s3 + (1, 0, 0).s3*coef.s2
    tmp = read_imagef(org_vol, sp, id+(int4)(1,0,0,0));
    ret.s23 += (tmp.s0+tmp.s1)*coef.s1;
    ret += tmp*coef.s2;


    // (0, 1, 1) += (0, 2, 0)*c1 : ret.s1 + (0, 1, 0).s0*coef.s1
    // (0, 1, 1) += (1, 2, 1)*c1 : ret.s1 + (0, 1, 0).s2*coef.s1
    // (1, 1, 0) += (0, 2, 0)*c1 : ret.s3 + (0, 1, 0).s0*coef.s1
    // (1, 1, 0) += (1, 2, 1)*c1 : ret.s3 + (0, 1, 0).s2*coef.s1
    
    // (0, 0, 0) += (0, 2, 0)*c2 : ret.s0 + (0, 1, 0).s0*coef.s2
    // (0, 1, 1) += (0, 3, 1)*c2 : ret.s1 + (0, 1, 0).s1*coef.s2
    // (1, 0, 1) += (1, 2, 1)*c2 : ret.s2 + (0, 1, 0).s2*coef.s2
    // (1, 1, 0) += (1, 3, 0)*c2 : ret.s3 + (0, 1, 0).s3*coef.s2
    tmp = read_imagef(org_vol, sp, id+(int4)(0,1,0,0));
    ret.s13 += (tmp.s0 + tmp.s2)*coef.s1;
    ret += tmp*coef.s2;


    // (0, 0, 0) += (0, -1, 1)*c1 : ret.s0 + (0, -1, 0).s1*coef.s1
    // (0, 0, 0) += (1, -1, 0)*c1 : ret.s0 + (0, -1, 0).s3*coef.s1
    // (1, 0, 1) += (0, -1, 1)*c1 : ret.s2 + (0, -1, 0).s1*coef.s1
    // (1, 0, 1) += (1, -1, 0)*c1 : ret.s2 + (0, -1, 0).s3*coef.s1
    
    // (0, 0, 0) += (0, -2, 0)*c2 : ret.s0 + (0, -1, 0).s0*coef.s2
    // (0, 1, 1) += (0, -1, 1)*c2 : ret.s1 + (0, -1, 0).s1*coef.s2
    // (1, 0, 1) += (1, -2, 1)*c2 : ret.s2 + (0, -1, 0).s2*coef.s2
    // (1, 1, 0) += (1, -1, 0)*c2 : ret.s3 + (0, -1, 0).s3*coef.s2
    tmp = read_imagef(org_vol, sp, id+(int4)(0,-1,0,0));
    ret.s02 += (tmp.s1 + tmp.s3)*coef.s1;
    ret += tmp*coef.s2;


    // (0, 0, 0) += (0, 1, -1)*c1 : ret.s0 + (0, 0, -1).s1*coef.s1
    // (0, 0, 0) += (1, 0, -1)*c1 : ret.s0 + (0, 0, -1).s2*coef.s1
    // (1, 1, 0) += (0, 1, -1)*c1 : ret.s3 + (0, 0, -1).s1*coef.s1
    // (1, 1, 0) += (1, 0, -1)*c1 : ret.s3 + (0, 0, -1).s2*coef.s1
    
    // (0, 0, 0) += (0, 0, -2)*c2 : ret.s0 + (0, 0, -1).s0*coef.s2
    // (0, 1, 1) += (0, 1, -1)*c2 : ret.s1 + (0, 0, -1).s1*coef.s2
    // (1, 0, 1) += (1, 0, -1)*c2 : ret.s2 + (0, 0, -1).s2*coef.s2
    // (1, 1, 0) += (1, 1, -2)*c2 : ret.s3 + (0, 0, -1).s3*coef.s2
    tmp = read_imagef(org_vol, sp, id+(int4)(0,0,-1,0));
    ret.s03 += (tmp.s1 + tmp.s2)*coef.s1;
    ret += tmp*coef.s2;

    // (0, 1, 1) += (0, 0, 2)*c1 : ret.s1 + (0, 0, 1).s0*coef.s1
    // (0, 1, 1) += (1, 1, 2)*c1 : ret.s1 + (0, 0, 1).s3*coef.s1
    // (1, 0, 1) += (0, 0, 2)*c1 : ret.s2 + (0, 0, 1).s0*coef.s1
    // (1, 0, 1) += (1, 1, 2)*c1 : ret.s2 + (0, 0, 1).s3*coef.s1

    // (0, 0, 0) += (0, 0, 2)*c2 : ret.s0 + (0, 0, 1).s0*coef.s2
    // (0, 1, 1) += (0, 1, 3)*c2 : ret.s1 + (0, 0, 1).s1*coef.s2
    // (1, 0, 1) += (1, 0, 3)*c2 : ret.s2 + (0, 0, 1).s2*coef.s2
    // (1, 1, 0) += (1, 1, 2)*c2 : ret.s3 + (0, 0, 1).s3*coef.s2
    tmp = read_imagef(org_vol, sp, id+(int4)(0,0,1,0));
    ret.s12 += (tmp.s0 + tmp.s3)*coef.s1;
    ret += tmp*coef.s2;

    // (1, 1, 0) += (2, 2, 0)*c1 : ret.s3 + (1, 1, 0).s0*coef.s1
    ret.s3 += read_imagef(org_vol, sp, id+(int4)(1,1,0,0)).s0*coef.s1;

    // (0, 1, 1) += (-1, 2, 1)*c1 : ret.s1 + (-1, 1, 0).s2*coef.s1
    ret.s1 += read_imagef(org_vol, sp, id+(int4)(-1,1,0,0)).s2*coef.s1;

    // (0, 0, 0) += (-1, -1, 0)*c1 : ret.s0 + (-1, -1, 0).s3*coef.s1
    ret.s0 += read_imagef(org_vol, sp, id+(int4)(-1,-1,0,0)).s3*coef.s1;

    // (1, 0, 1) += (2, -1, 1)*c1 : ret.s2 + (1, -1, 0).s1*coef.s1
    ret.s2 += read_imagef(org_vol, sp, id+(int4)(1,-1,0,0)).s1*coef.s1;

    // (1, 0, 1) += (2, 0, 2)*c1 : ret.s2 + (1, 0, 1).s0*coef.s1
    ret.s2 += read_imagef(org_vol, sp, id+(int4)(1,0,1,0)).s0*coef.s1;

    // (1, 1, 0) += (2, 1, -1)*c1 : ret.s3 + (1, 0, -1).s1*coef.s1
    ret.s3 += read_imagef(org_vol, sp, id+(int4)(1,0,-1,0)).s1*coef.s1;

    // (0, 0, 0) += (-1, 0, -1)*c1 : ret.s0 + (-1, 0, -1).s2*coef.s1
    ret.s0 += read_imagef(org_vol, sp, id+(int4)(-1,0,-1,0)).s2*coef.s1;

    // (0, 1, 1) += (-1, 1, 2)*c1 : ret.s1 + (-1, 0, 1).s3*coef.s1
    ret.s1 += read_imagef(org_vol, sp, id+(int4)(-1,0,1,0)).s3*coef.s1;

    // (0, 1, 1) += (0, 2, 2)*c1 : ret.s1 + (0, 1, 1).s0*coef.s1
    ret.s1 += read_imagef(org_vol, sp, id+(int4)(0,1,1,0)).s0*coef.s1;

    // (1, 0, 1) += (1, -1, 2)*c1 : ret.s2 + (0, -1, 1).s3*coef.s1
    ret.s2 += read_imagef(org_vol, sp, id+(int4)(0,-1,1,0)).s3*coef.s1;

    // (0, 0, 0) += (0, -1, -1)*c1 : ret.s0 + (0, -1, -1).s1*coef.s1
    ret.s0 += read_imagef(org_vol, sp, id+(int4)(0,-1,-1,0)).s1*coef.s1;

    // (1, 1, 0) += (1, 2, -1)*c1 : ret.s3 + (0, 1, -1).s2*coef.s1
    ret.s3 += read_imagef(org_vol, sp, id+(int4)(0,1,-1,0)).s2*coef.s1;


    write_imagef(vol, id, ret);

}