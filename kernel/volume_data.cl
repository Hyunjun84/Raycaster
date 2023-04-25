#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

const sampler_t sp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void applyQuasiInterpolator_CC(__write_only image3d_t vol, __read_only image3d_t org_vol, float4 coef, int4 dim)
{
    int4 id = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 1);
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

__inline int4 idx2id(int idx, int4 sz)
{
    int4 id;
    id.x = idx%sz.x;
    idx /= sz.x;
    id.y = idx%sz.y;
    id.z = idx/sz.y;
    id.w = 0;
    return id;
}

__inline int id2idx(int4 id, int4 sz)
{
    return (id.z*sz.y + id.y)*sz.x + id.x;
}
__kernel void applyQuasiInterpolator_CC_loc(__write_only image3d_t vol, __read_only image3d_t org_vol, float4 coef, int4 dim)
{
    int4 id = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 1);
    int4 lid = (int4)(get_local_id(0), get_local_id(1), get_local_id(2), 1);
    int4 lsiz = (int4)(get_local_size(0), get_local_size(1), get_local_size(2), 1);
    int lidx = id2idx(lid, lsiz);

    // extended size
    int4 esiz = (int4)(lsiz.x+2, lsiz.y+2, lsiz.z+2,1);

    // group origin
    int4 gorg = (int4)(get_group_id(0), get_group_id(1), get_group_id(2), 1)*lsiz-1;

    // 6^3=216
    __local float buf[216];

    for(int i=0; i<216; i+=64) {
        int tidx = lidx+i;
        if(tidx>=216) break;
        int4 tid = idx2id(tidx, esiz);
        buf[tidx] = read_imagef(org_vol, sp, tid+gorg).x;
    } 
    barrier(CLK_LOCAL_MEM_FENCE);

#if 0
    lid += 1; 
    float val = buf[id2idx(lid, esiz)]*coef.s0;

    // First neighbor
    val += buf[id2idx(lid + (int4)(1,0,0,0), esiz)]*coef.s1;
    val += buf[id2idx(lid + (int4)(0,1,0,0), esiz)]*coef.s1;
    val += buf[id2idx(lid + (int4)(0,0,1,0), esiz)]*coef.s1;
    val += buf[id2idx(lid + (int4)(-1,0,0,0), esiz)]*coef.s1;
    val += buf[id2idx(lid + (int4)(0,-1,0,0), esiz)]*coef.s1;
    val += buf[id2idx(lid + (int4)(0,0,-1,0), esiz)]*coef.s1;

    // Second neighbor
    val += buf[id2idx(lid + (int4)(0,1,1,0), esiz)]*coef.s2;
    val += buf[id2idx(lid + (int4)(1,0,1,0), esiz)]*coef.s2;
    val += buf[id2idx(lid + (int4)(1,1,0,0), esiz)]*coef.s2;
    val += buf[id2idx(lid + (int4)( 0,-1,1,0), esiz)]*coef.s2;
    val += buf[id2idx(lid + (int4)(-1, 0,1,0), esiz)]*coef.s2;
    val += buf[id2idx(lid + (int4)(-1, 1,0,0), esiz)]*coef.s2;
    val += buf[id2idx(lid + (int4)(0, 1,-1,0), esiz)]*coef.s2;
    val += buf[id2idx(lid + (int4)(1, 0,-1,0), esiz)]*coef.s2;
    val += buf[id2idx(lid + (int4)(1,-1, 0,0), esiz)]*coef.s2;
    val += buf[id2idx(lid + (int4)( 0,-1,-1,0), esiz)]*coef.s2;
    val += buf[id2idx(lid + (int4)(-1, 0,-1,0), esiz)]*coef.s2;
    val += buf[id2idx(lid + (int4)(-1,-1, 0,0), esiz)]*coef.s2;
#else
    lidx = id2idx(lid, esiz);
    float val = buf[lidx+43]*coef.s0;

    // First neighbor
    val += buf[lidx + 44]*coef.s1;
    val += buf[lidx + 49]*coef.s1;
    val += buf[lidx + 43]*coef.s1;
    val += buf[lidx + 42]*coef.s1;
    val += buf[lidx + 37]*coef.s1;
    val += buf[lidx + 7]*coef.s1;

    // Second neighbor
    val += buf[lidx + 85]*coef.s2;
    val += buf[lidx + 80]*coef.s2;
    val += buf[lidx + 50]*coef.s2;
    val += buf[lidx + 73]*coef.s2;
    val += buf[lidx + 78]*coef.s2;
    val += buf[lidx + 48]*coef.s2;
    val += buf[lidx + 13]*coef.s2;
    val += buf[lidx + 8]*coef.s2;
    val += buf[lidx + 38]*coef.s2;
    val += buf[lidx + 1]*coef.s2;
    val += buf[lidx + 6]*coef.s2;
    val += buf[lidx + 36]*coef.s2;
#endif
    write_imagef(vol, id, val);
}


__kernel void applyQuasiInterpolator_FCC(__write_only image3d_t vol, __read_only image3d_t org_vol, float4 coef, int4 dim)
{
    int4 id = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 1);
    
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


__kernel void applyQuasiInterpolator_FCC_loc(__write_only image3d_t vol, __read_only image3d_t org_vol, float4 coef, int4 dim)
{
    int4 id = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 1);
    int4 lid = (int4)(get_local_id(0), get_local_id(1), get_local_id(2), 1);
    int4 lsiz = (int4)(get_local_size(0), get_local_size(1), get_local_size(2), 1);
    int lidx = id2idx(lid, lsiz);

    // extended size
    int4 esiz = (int4)(lsiz.x+2, lsiz.y+2, lsiz.z+2,1);

    // group origin
    int4 gorg = (int4)(get_group_id(0), get_group_id(1), get_group_id(2), 1)*lsiz-1;

    //  6^3=216 local memory
    __local float4 buf[216];

    for(int i=0; i<216; i+=64) {
        int tidx = lidx+i;
        if(tidx>=216) break;
        int4 tid = idx2id(tidx, esiz);
        buf[tidx] = read_imagef(org_vol, sp, tid+gorg);
    } 
    barrier(CLK_LOCAL_MEM_FENCE);

    if(any(id.xyz>=dim.xyz)) return;

#if 0
    lid += 1; 
    float4 tmp = buf[id2idx(lid, esiz)];
    float4 ret = tmp*coef.s0;
    ret += (tmp.s1230 + tmp.s2301 + tmp.s3012)*coef.s1;

    tmp = buf[id2idx(lid+(int4)(-1,0,0,0), esiz)];
    ret.s01 += (tmp.s2+tmp.s3)*coef.s1;
    ret += tmp*coef.s2;

    tmp = buf[id2idx(lid+(int4)(1,0,0,0), esiz)];
    ret.s23 += (tmp.s0+tmp.s1)*coef.s1;
    ret += tmp*coef.s2;

    tmp = buf[id2idx(lid+(int4)(0,1,0,0), esiz)];
    ret.s13 += (tmp.s0 + tmp.s2)*coef.s1;
    ret += tmp*coef.s2;

    tmp = buf[id2idx(lid+(int4)(0,-1,0,0), esiz)];
    ret.s02 += (tmp.s1 + tmp.s3)*coef.s1;
    ret += tmp*coef.s2;

    tmp = buf[id2idx(lid+(int4)(0,0,-1,0), esiz)];
    ret.s03 += (tmp.s1 + tmp.s2)*coef.s1;
    ret += tmp*coef.s2;

    tmp = buf[id2idx(lid+(int4)(0,0,1,0), esiz)];
    ret.s12 += (tmp.s0 + tmp.s3)*coef.s1;
    ret += tmp*coef.s2;

    ret.s3 += buf[id2idx(lid+(int4)(1,1,0,0), esiz)].s0*coef.s1;
    ret.s1 += buf[id2idx(lid+(int4)(-1,1,0,0), esiz)].s2*coef.s1;
    ret.s0 += buf[id2idx(lid+(int4)(-1,-1,0,0), esiz)].s3*coef.s1;
    ret.s2 += buf[id2idx(lid+(int4)(1,-1,0,0), esiz)].s1*coef.s1;
    ret.s2 += buf[id2idx(lid+(int4)(1,0,1,0), esiz)].s0*coef.s1;
    ret.s3 += buf[id2idx(lid+(int4)(1,0,-1,0), esiz)].s1*coef.s1;
    ret.s0 += buf[id2idx(lid+(int4)(-1,0,-1,0), esiz)].s2*coef.s1;
    ret.s1 += buf[id2idx(lid+(int4)(-1,0,1,0), esiz)].s3*coef.s1;
    ret.s1 += buf[id2idx(lid+(int4)(0,1,1,0), esiz)].s0*coef.s1;
    ret.s2 += buf[id2idx(lid+(int4)(0,-1,1,0), esiz)].s3*coef.s1;
    ret.s0 += buf[id2idx(lid+(int4)(0,-1,-1,0), esiz)].s1*coef.s1;
    ret.s3 += buf[id2idx(lid+(int4)(0,1,-1,0), esiz)].s2*coef.s1;
#else
    lidx = id2idx(lid, esiz);
    float4 tmp = buf[lidx+43];
    float4 ret = tmp*coef.s0;
    ret += (tmp.s1230 + tmp.s2301 + tmp.s3012)*coef.s1;

    tmp = buf[lidx + 42];
    ret.s01 += (tmp.s2+tmp.s3)*coef.s1;
    ret += tmp*coef.s2;

    tmp = buf[lidx + 44];
    ret.s23 += (tmp.s0+tmp.s1)*coef.s1;
    ret += tmp*coef.s2;

    tmp = buf[lidx + 49];
    ret.s13 += (tmp.s0 + tmp.s2)*coef.s1;
    ret += tmp*coef.s2;

    tmp = buf[lidx + 37];
    ret.s02 += (tmp.s1 + tmp.s3)*coef.s1;
    ret += tmp*coef.s2;

    tmp = buf[lidx + 7];
    ret.s03 += (tmp.s1 + tmp.s2)*coef.s1;
    ret += tmp*coef.s2;

    tmp = buf[lidx + 79];
    ret.s12 += (tmp.s0 + tmp.s3)*coef.s1;
    ret += tmp*coef.s2;

    ret.s3 += buf[lidx + 50].s0*coef.s1;
    ret.s1 += buf[lidx + 48].s2*coef.s1;
    ret.s0 += buf[lidx + 36].s3*coef.s1;
    ret.s2 += buf[lidx + 38].s1*coef.s1;
    ret.s2 += buf[lidx + 80].s0*coef.s1;
    ret.s3 += buf[lidx + 8].s1*coef.s1;
    ret.s0 += buf[lidx + 5].s2*coef.s1;
    ret.s1 += buf[lidx + 78].s3*coef.s1;
    ret.s1 += buf[lidx + 85].s0*coef.s1;
    ret.s2 += buf[lidx + 73].s3*coef.s1;
    ret.s0 += buf[lidx + 1].s1*coef.s1;
    ret.s3 += buf[lidx + 13].s2*coef.s1;
#endif
    write_imagef(vol, id, ret);
}

/*
    gen minmax buffer
    each level has dobluing
        1^3 ... maximum pot lesser than max(dim/2)^3
    mimmax buffer size is
        sum_{n=1}^k = 8/7*(2^{3*k}-1)+1
*/
__kernel void genQuasiMinMaxBuffer_test(__write_only image3d_t out, __write_only image3d_t in, __local float* buf, int8 stencil)
{
    int3 gid = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    int3 lid = (int3)(get_local_id(0), get_local_id(1), get_local_id(2));
    int3 Gid = (int3)(get_group_id(0), get_group_id(1), get_group_id(2));

    int3 lsiz = (int3)(get_local_size(0), get_local_size(1), get_local_size(2));
    int lidx = lsiz.x*(lsiz.y*lid.z + lid.y) + lid.x;









}
