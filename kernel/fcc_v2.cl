
#define SCALE_F (0.041666667f)

__inline int sgn(float x) 
{
    return convert_int(x>=0.0f)*2-1;
}

__inline float eval(float3 p_in, __read_only image3d_t vol)
{
    // Find origin
    int3 org = convert_int3(round(p_in));
    if( (convert_int(org.x+org.y+org.z)&0x01) != 0 ) {
        float3 l = p_in-convert_float3(org);
        float3 ul = fabs(l);
        if(ul.x > ul.y) {
            if(ul.x > ul.z)   org.x += sgn(l.x);
            else              org.z += sgn(l.z);
        } else {
            if(ul.y > ul.z)   org.y += sgn(l.y);
            else              org.z += sgn(l.z);
        }
    }
    float3 p_local = p_in - convert_float3(org);    // local coordinates

    // computes the membership against the six knot planes intersecting the unit cube centered at the local origin
    int dR[6] = {  
        convert_int(p_local.z>=p_local.y),
        convert_int(p_local.z>=p_local.x),
        convert_int(p_local.y>=p_local.x),
        convert_int(p_local.x>=-p_local.y),
        convert_int(p_local.x>=-p_local.z),
        convert_int(p_local.y>=-p_local.z)
    };

    // type_R: the `reflection transformation' which is one of four `even reflections'
    // The reflection matrix R for each type:
    // (0,0,0): [ 1, 0, 0] (0,1,1): [ 1, 0, 0] (1,0,1): [-1, 0, 0] (1,1,0): [-1, 0, 0]
    //          [ 0, 1, 0]          [ 0,-1, 0]          [ 0, 1, 0]          [ 0,-1, 0]
    //          [ 0, 0, 1]          [ 0, 0,-1]          [ 0, 0,-1]          [ 0, 0, 1]
    int3 R = (int3)((1-dR[1])*(1-dR[2])*(1-dR[5]),
                    (1-dR[0])*   dR[2] *(1-dR[4]),
                       dR[0] *   dR[1] *(1-dR[3]));
    R += dR[3]*dR[4]*dR[5] - R.yzx - R.zxy;
    
    // Transform p_local into the `reference left coset' (Fig 2(a)) hit_zx the reflection computed above.
    // Same as R^-1*p_local (R is one of the reflection matrices defined above)
    // Note that R^{-1}=R since R is symmetric & orthogonal.
    float3  p_ref_R = convert_float3(R)*p_local;

    // Compute the membership against the three knot planes intersecting the piece in Fig 2(a).
    // Three knot planes with their normals (-1,1,0), (-1,0,1), and (0,-1,1), respectively.
    // The input (p_ref_R) belong to one of the six tetrahedra in Fig 2(a)
    // and each piece corresponds to one of the six permutation matrices P below.
    int3 dP = (int3)(convert_int(p_ref_R.y>=p_ref_R.x),    
                     convert_int(p_ref_R.z>=p_ref_R.x), 
                     convert_int(p_ref_R.z>=p_ref_R.y));

    #if 0    
    dP = (int3)(dP.x+dP.y, dP.x-dP.z, dP.y+dP.z);
    int3 vecPx = (int3)(dP.x==0, dP.y==1, dP.z==2);
    int3 vecPy = (int3)(dP.x==1, dP.y==0, dP.z==1);
    int3 vecPz = (int3)(dP.x==2, dP.y==-1,dP.z==0);
    #else
    int idx_P = 2*(dP.x+dP.y)+dP.z;
    int3 vecP1 = (int3)(convert_int(idx_P==0), convert_int(idx_P==4), convert_int(idx_P==3));
    int3 vecP2 = (int3)(convert_int(idx_P==1), convert_int(idx_P==2), convert_int(idx_P==5));

    int3 vecPx = vecP1+vecP2;
    int3 vecPy = vecP1.zxy+vecP2.yzx;
    int3 vecPz = vecP1.yzx+vecP2.zxy;
    #endif

    
    // Compute the permutation matrix P from type_P.
    // (0,0,0):[1,0,0] (0,0,1):[1,0,0] (1,0,0):[0,1,0] (0,1,1):[0,0,1] (1,1,0):[0,1,0] (1,1,1):[0,0,1]
    //         [0,1,0]         [0,0,1]         [1,0,0]         [1,0,0]         [0,0,1]         [0,1,0]
    //         [0,0,1]         [0,1,0]         [0,0,1]         [0,1,0]         [1,0,0]         [1,0,0]
    // For p_ref_R in one of the six tetrahedral pieces, P^{-1}*p_ref_R is inside the reference tetrahedron.
    // Note that mat3 is in column-major format.

    // Transform p_ref_R into the `reference tetrahedron' hit_zx multiplying P.
    float3 p_ref;

    p_ref.x = dot(p_ref_R, convert_float3(vecPx));
    p_ref.y = dot(p_ref_R, convert_float3(vecPy));
    p_ref.z = dot(p_ref_R, convert_float3(vecPz));

    // Compute the barycentric coordinates.
    float4 u1 = (float4)(1.0f-p_ref.x-p_ref.y, 
                        p_ref.y+p_ref.z, 
                        p_ref.x-p_ref.y, 
                        p_ref.y-p_ref.z);
    // Fetch coefficients

    int3 dirx = R*vecPx;
    int3 diry = R*vecPy;
    int3 dirz = R*vecPz;

    // We re-ordered the fetches such that
    // the displacement of adjacent fetches are axis-aligned.
    // Therefore, we can move to the next coefficient with low computational overhead.
    float c0 = read_imagef(vol, sp2, (int4)(org,1)).r;
    float c1 = read_imagef(vol, sp2, (int4)(org+2*dirx,1)).r;
    float c2 = read_imagef(vol, sp2, (int4)(org+diry-dirz,1)).r;
    float c3 = read_imagef(vol, sp2, (int4)(org+diry+dirz,1)).r;
    float c4 = read_imagef(vol, sp2, (int4)(org+dirx-dirz,1)).r;
    float c5 = read_imagef(vol, sp2, (int4)(org+dirx+dirz,1)).r;
    float c6 = read_imagef(vol, sp2, (int4)(org+dirx-diry,1)).r;
    float c7 = read_imagef(vol, sp2, (int4)(org+dirx+diry,1)).r;


    // Evaluation

    float val = 0;
    float4 u2 = u1*u1;
    float4 u3 = u2*u1;
    
    float c07 = c0+c7;
    float c16 = c1+c6;
    float c0247 = c07+c2+c4;
    float c0357 = c07+c3+c5;
    float c4567 = c4+c5+c6+c7;

    val += u2.x*(u1.y*(8*c0+c0357) +
                 u1.z*(8*c0+c4567) +
                 u1.w*(8*c0+c0247));

    val += u1.x*(2*(u2.y*(2*c0+c0357) +
                    u2.z*(2*c0+c4567) +
                    u2.w*(2*c0+c0247)) +
                u1.z*(u1.w*(4*(c4+c7) + (12*c0 + c4567))+
                      u1.y*(4*(c5+c7) + (12*c0 + c4567)))+
                u1.y*u1.w*(4*c7+c0247+(12*c0 + c0357)));
    
    val += u3.y*c0357 + 
           u3.w*c0247 +
           u1.y*u1.w*(u1.y*(4*c07+c0357) +
                      u1.w*(4*c07+c0247)) + 
           u2.z*(u1.y*(3*(c0357-c3) + (c16+c4))+
                 u1.w*(3*(c0247-c2) + (c16+c5)));

    val += 4*(u1.z*(u2.y*(c0357-c3)+u2.w*(c0247-c2)) + (u3.x*c0));
    val += u1.y*u1.z*u1.w*(2*(c4+c5) + (9*c07+c16));
    val *= 6;
    val += 4*u3.z*(c0+c1+c4567);
    
    
    return val*SCALE_F;
}

