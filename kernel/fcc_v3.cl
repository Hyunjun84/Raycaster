
#define SCALE_F (0.000086806f)   // 1/11520
#define SCALE_G (0.002604167f)   // 1/15336

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
    
    // Fetch coefficients
    int3 dir2x = 2*R*vecPx;
    int3 dir2y = 2*R*vecPy;
    int3 dir2z = 2*R*vecPz;

    // We re-ordered the fetches such that
    // the displacement of adjacent fetches are axis-aligned.
    // Therefore, we can move to the next coefficient with low computational overhead.
    int3 offset = org;              float c0 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 0, 0, 0)
    offset -= dir2z;                float c15 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 0,-2)
    offset += (dir2x+dir2y)>>1;     float c19 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 1,-2)
    offset += 2*dir2z;              float c20 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 1, 2)
    offset -= (dir2x+dir2y)>>1;     float c16 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 0, 2)
    offset -= (dir2y+dir2z);        float c14 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0,-2, 0)
    offset += 2*dir2y;              float c17 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 2, 0)
    offset += (dir2x-dir2z)>>1;     float c21 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 2,-1)
    offset += dir2z;                float c22 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 2, 1)
    offset -= dir2y;                float c11 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 0, 1)
    offset -= dir2x;                float c3 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //(-1, 0, 1)
    offset -= dir2z;                float c2 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //(-1, 0,-1)
    offset += dir2x;                float c10 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 0,-1)
    offset += (dir2y+dir2z)>>1;     float c12 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 1, 0)
    offset -= dir2x;                float c4 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //(-1, 1, 0)
    offset -= dir2y;                float c1 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //(-1,-1, 0)
    offset += dir2x;                float c9 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 1,-1, 0)
    offset -= (dir2x+dir2z)>>1;     float c5 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 0,-1,-1)
    offset += dir2z;                float c6 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 0,-1, 1)
    offset += (dir2y-dir2z);        float c7 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 0, 1,-1)
    offset += dir2z;                float c8 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 0, 1, 1)
    offset += dir2x;                float c26 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2, 1, 1)
    offset -= dir2z;                float c25 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2, 1,-1)
    offset += (dir2z-dir2y);        float c24 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2,-1, 1)
    offset -= dir2z;                float c23 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2,-1,-1)
    offset += (dir2y+dir2z)>>1;     float c18 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2, 0, 0)
    offset -= 2*dir2x;              float c13 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //(-2, 0, 0)

    // Evaluation
    float val = 0;
    float3 X = p_ref;
    float3 X2 = X*X;
    float3 X3 = X2*X;
    float3 X4 = X3*X;
    float3 X5 = X4*X;
    float3 X6 = X5*X;
    
    val +=    9.f*(X6.y*(c19+c20+c25+c26-c2-c3-c15-c16+2.f*(c9-c1-c18-c21-c22)+4.f*(c4+c14)-5.f*(c10+c11)+7.f*(c7+c8)-8.f*(c0+c17)+10.f*c12)+
                 X6.z*((c2+c3+c5+c6+c21+c22+c25+c26)-2.f*(c4+c9+c15+c16+c17+c18+c19+c20)+6.f*(c7+c8+c10+c11)-8.f*(c0+c12)));
    
    val +=   18.f*(X6.x*((c9+c10+c11+c12-c1-c2-c3-c4)+2.f*(c13-c18))+
                 X.x*(X5.y*((-c25-c26)+2.f*(c13+c15+c16-c19-c20)+3.f*(-c5-c6+c21+c22)+4.f*(-c0-c4-c10-c11-c18)+5.f*(c2+c3)+6.f*(-c1-c17)+10.f*(c12)+12.f*(c9))+
                      X5.z*((-c2+c3-c7+c8+c10-c11+c25-c26)+2.f*(-c5+c6+c21-c22)+3.f*(c15-c16-c19+c20)))+
                 X5.y*(X.z*((c19-c20)+2.f*(c25-c26)+3.f*(c10-c11+c15-c16-c21+c22)+6.f*(c2-c3)+7.f*(-c7+c8)+9.f*(-c5+c6))+
                       ((-c15-c16)+2.f*(-c10-c11-c13)+3.f*(c7+c8)+6.f*(-c2-c3+c4+c9)+9.f*(c5+c6)+12.f*(c1-c14)+16.f*(-c0)))+
                 X.y*X5.z*((-c5+c6+c7-c8-c10+c11+c21-c22)+2.f*(-c2+c3+c25-c26)+3.f*(c15-c16-c19+c20)));
    
    val +=   36.f*(X5.x*(X.y*((c14-c17)+2.f*(c5+c6-c7-c8)+3.f*(-c1+c4-c9+c12))+
                       X.z*((c15-c16)+2.f*(c5-c6+c7-c8)+3.f*(-c2+c3-c10+c11))+
                       (-c14-c15-c16-c17)+4.f*(-c5-c6-c7-c8)+6.f*(c1+c2+c3+c4+c9+c10+c11+c12-c13-c18)+16.f*(-c0))+
                 X.x*(6.f*(-c13+c18)+37.f*(-c1-c2-c3-c4+c9+c10+c11+c12))+
                 X.y*(6.f*(-c14+c17)+37.f*(-c1+c4-c5-c6+c7+c8-c9+c12))+
                 X.z*(6.f*(-c15+c16)+37.f*(-c2+c3-c5+c6-c7+c8-c10+c11))+
                 (c13+c14+c15+c16+c17+c18)+14.f*(c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+c11+c12)+146.f*c0);
    
    val +=   45.f*(X2.x*(X4.y*((c5+c6+c7+c8+c15+c16+c21+c22)+2.f*(c1+c2+c3+c13-c14+c18+c23+c24)+3.f*(-c19-c20)+4.f*(c17-c25-c26)+5.f*(c10+c11)+6.f*(c12)+8.f*(-c0-c9)+10.f*(-c4))+
                       X4.z*((-c10-c11+c15+c16+c19+c20)+2.f*(c13+c14+c18+c23+c24)+3.f*(-c5-c6-c21-c22)+4.f*(-c2-c3-c4-c25-c26)+5.f*(c7+c8)+6.f*(-c9)+8.f*(-c0+c1)+16.f*(c12)))+
                 X4.y*(X2.z*((c2+c3+c21+c22)+2.f*(-c7-c8+c13-c14+c15+c16+c23+c24)+3.f*(c5+c6-c25-c26)+4.f*(c17-c19-c20)+6.f*(-c4+c10+c11)+8.f*(-c0+c12)+10.f*(-c9))+
                       (2.f*(c13+c15+c16+c18+c21+c22)+9.f*(c2+c3+c10+c11)+12.f*(-c4-c9+c14+c17)+13.f*(-c5-c6)+14.f*(-c1)+15.f*(-c7-c8)+18.f*(-c12)+40.f*(c0)))+
                 X2.y*X4.z*((-c10-c11+c15+c16+c19+c20)+2.f*(c13+c14+c17+c23+c24)+3.f*(-c2-c3-c25-c26)+4.f*(-c5-c6-c9-c21-c22)+5.f*(c7+c8)+6.f*(-c4)+8.f*(-c0+c1)+16.f*(c12))+
                 X4.z*(2.f*(c13+c14+c17+c18+c19+c20)+8.f*(c1+c12)+10.f*(c4+c9)+12.f*(c15+c16)+13.f*(-c2-c3-c5-c6)+15.f*(-c7-c8-c10-c11)+40.f*(c0)));
    
    val +=   54.f*X5.z*((c2-c3+c5-c6+c7-c8+c10-c11)+2.f*(-c15+c16));
    
    val +=   90.f*(X4.x*(X2.y*((c1+c4+c5+c6+c7+c8-c13+c14+c15+c16+c17+c23+c24+c25+c26)+2.f*(c10+c11-c19-c20-c21-c22)+3.f*(-c18)+5.f*(-c9)+7.f*(c12)+8.f*(-c0))+
                       X2.z*((c2+c3+c5+c6+c7+c8+c10+c11-c13+c14+c15+c16+c17+c23+c24+c25+c26)+2.f*(-c19-c20-c21-c22)+3.f*(-c18)+4.f*(-c9)+8.f*(-c0+c12))+
                       (c14+c15+c16+c17+c23+c24+c25+c26)+5.f*(c5+c6+c7+c8)+6.f*(c13+c18)+7.f*(-c1-c2-c3-c4)+9.f*(-c9-c10-c11-c12)+24.f*(c0))+
                 X2.x*(X2.y*(X2.z*((c5+c6)+2.f*(-c2-c3)+4.f*(c13+c14)+7.f*(c15+c16)+8.f*(-c18-c25-c26)+10.f*(c17+c23+c24)+11.f*(-c19-c20-c21-c22)+13.f*(c7+c8+c10+c11)+16.f*(c1)+20.f*(-c4)+38.f*(-c9)+56.f*(-c0)+64.f*(c12))+
                            ((c19+c20+c21+c22+c23+c24+c25+c26)+4.f*(c15+c16)+5.f*(-c5-c6-c7-c8-c10-c11)+6.f*(c12-c13-c14-c17-c18)+8.f*(-c2-c3)+12.f*(c9)+18.f*(c1+c4)))+
                       X2.z*((c19+c20+c21+c22+c23+c24+c25+c26)+2.f*(-c9)+4.f*(c14+c17)+5.f*(-c5-c6-c7-c8)+6.f*(-c13-c15-c16-c18)+8.f*(-c1-c4-c12)+9.f*(c10+c11)+18.f*(c2+c3))+
                       (-c14-c15-c16-c17)+6.f*(c13+c18)+8.f*(-c5-c6-c7-c8)+10.f*(c1+c2+c3+c4+c9+c10+c11+c12)+56.f*(-c0))+
                 X.x*(X4.y*(X.z*((-c15+c16)+2.f*(c5-c6-c23+c24)+3.f*(-c10+c11+c19-c20+c25-c26)+5.f*(c2-c3)+9.f*(-c7+c8))+
                           ((-c15-c16)+2.f*(c5+c6+c7+c8-c13+c18+c21+c22)+3.f*(c10+c11)+4.f*(c1)+5.f*(-c2-c3)+6.f*(-c9-c17)+8.f*(-c12)+10.f*(c4)))+
                      X.y*X4.z*(2.f*(-c4-c9+c13+c14-c23-c24)+3.f*(-c2-c3-c5-c6-c21-c22-c25-c26)+4.f*(c7+c8+c12+c17+c18)+8.f*(c1)+10.f*(c10+c11)+20.f*(-c0))+
                      X4.z*(2.f*(-c4-c7-c8+c9-c13-c14+c18+c19+c20)+3.f*(-c15-c16)+4.f*(c12)+6.f*(c5+c6)+7.f*(c2+c3-c10-c11)+8.f*(-c1)))+
                 X4.y*X.z*(2.f*(-c15+c16-c21+c22)+3.f*(-c10+c11)+5.f*(-c2+c3+c5-c6)+9.f*(c7-c8))+
                 X2.y*(X2.z*((c19+c20+c21+c22+c23+c24+c25+c26)+2.f*(-c4)+4.f*(c13+c18)+5.f*(-c2-c3)+6.f*(-c14-c15-c16-c17)+8.f*(-c1-c9-c10-c11-c12)+12.f*(c7+c8)+18.f*(c5+c6))+
                      ((-c13-c15-c16-c18)+6.f*(c14+c17)+8.f*(-c2-c3-c10-c11)+10.f*(c1+c4+c5+c6+c7+c8+c9+c12)+56.f*(-c0)))+
                 X.y*X4.z*(2.f*(c4-c9-c10-c11-c13-c14+c17+c19+c20)+3.f*(-c15-c16)+4.f*(c12)+6.f*(c2+c3)+7.f*(c5+c6-c7-c8)+8.f*(-c1))+
                 X2.z*((-c13-c14-c17-c18)+6.f*(c15+c16)+8.f*(-c1-c4-c9-c12)+10.f*(c2+c3+c5+c6+c7+c8+c10+c11)+56.f*(-c0)));
    
    val +=  180.f*(X4.x*(X.y*(X.z*((c23-c24-c25+c26)+2.f*(c19-c20+c21-c22)+3.f*(c5-c6-c7+c8)+6.f*(-c10+c11))+
                                (-c14+c17-c23-c24+c25+c26)+2.f*(c1-c4-c5-c6+c7+c8)+6.f*(c9-c12))+
                       X.z*((-c15+c16-c23+c24-c25+c26)+2.f*(c2-c3-c5+c6-c7+c8)+6.f*(c10-c11)))+
                 X3.x*(X3.y*((c10+c11+c15+c16-c19-c20-c21-c22-c23-c24-c25-c26)+2.f*(c1-c4+c7+c8+c12+c17)+4.f*(c18)+8.f*(-c0))+
                       X2.y*(X.z*((-c15+c16)+3.f*(c19-c20+c21-c22-c23+c24+c25-c26)+4.f*(c2-c3)+5.f*(-c10+c11)+6.f*(c5-c6)+12.f*(-c7+c8))+
                                 (-c10-c11+c19+c20+c21+c22)+2.f*(-c9+c13-c15-c16+c23+c24+c25+c26)+4.f*(-c1-c4)+6.f*(-c18)+8.f*(-c12)+16.f*(c0))+
                       X.y*(X2.z*(2.f*(-c9+c14)+3.f*(c10+c11+c15+c16-c19-c20-c21-c22-c23-c24-c25-c26)+4.f*(c1-c4+c17)+6.f*(c7+c8)+8.f*(c12)+12.f*(c18)+24.f*(-c0))+
                                 (-c5-c6+c7+c8-c23-c24+c25+c26)+4.f*(c9-c12))+
                       X3.z*((-c10+c11-c15+c16+c19-c20+c21-c22-c23+c24+c25-c26)+2.f*(c2-c3+c5-c6)+4.f*(-c7+c8))+
                       X2.z*((c19+c20+c21+c22)+2.f*(c9+c13-c14-c17+c23+c24+c25+c26)+4.f*(-c2-c3-c12)+5.f*(-c10-c11)+6.f*(-c18)+16.f*(c0))+
                       X.z*((-c5+c6-c7+c8-c23+c24-c25+c26)+4.f*(c10-c11)))+
                 X2.x*(X3.y*(X.z*(2.f*(c21-c22)+3.f*(c5-c6+c23-c24+c25-c26)+4.f*(c19-c20)+6.f*(c2-c3)+9.f*(-c7+c8)+16.f*(-c10+c11))+
                            ((-c5-c6-c7-c8-c15-c16+c19+c20-c23-c24+c25+c26)+2.f*(c9+c14+c21+c22)+4.f*(-c1+c4-c17)+8.f*(c0-c12)))+
                       X2.y*X.z*((-c15+c16)+4.f*(-c19+c20)+5.f*(-c5+c6-c21+c22-c23+c24-c25+c26)+8.f*(-c2+c3)+13.f*(c7-c8)+25.f*(c10-c11))+
                       X.y*(X3.z*(2.f*(-c15+c16+c19-c20)+3.f*(c23-c24+c25-c26)+4.f*(c21-c22)+5.f*(c5-c6)+6.f*(c2-c3)+7.f*(-c7+c8)+14.f*(-c10+c11))+
                            X2.z*((c5+c6)+2.f*(-c17)+3.f*(-c15-c16)+4.f*(-c14+c21+c22)+5.f*(c19+c20-c23-c24+c25+c26)+6.f*(-c10-c11)+7.f*(-c7-c8)+8.f*(-c1+c4)+22.f*(c9)+24.f*(c0)+28.f*(-c12))+
                            X.z*((-c19+c20-c21+c22)+2.f*(c23-c24-c25+c26)+3.f*(c10-c11)+4.f*(-c5+c6+c7-c8)))+
                       X3.z*((-c21+c22-c23+c24-c25+c26)+2.f*(-c19+c20)+3.f*(-c5+c6+c7-c8+c15-c16)+4.f*(-c2+c3)+5.f*(c10-c11)))+
                 X.x*(X3.y*(X2.z*((c2+c3-c5-c6)+2.f*(c13-c21-c22)+3.f*(c15+c16-c19-c20-c23-c24)+4.f*(c1+c7+c8+c9+c17-c25-c26)+5.f*(c10+c11)+6.f*(-c4)+8.f*(c18)+12.f*(c12)+28.f*(-c0))+
                            ((-c5-c6-c7-c8+c10+c11+c21+c22)+4.f*(c0-c12)))+
                      X2.y*(X3.z*(2.f*(-c15+c16+c19-c20)+3.f*(c21-c22-c23+c24)+4.f*(-c10+c11+c25-c26)+5.f*(c2-c3)+6.f*(c5-c6)+11.f*(-c7+c8))+
                            X2.z*((c2+c3)+3.f*(-c15-c16)+4.f*(-c9-c13+c23+c24+c25+c26)+5.f*(c19+c20+c21+c22)+6.f*(-c17)+8.f*(-c1-c18)+10.f*(c4-c10-c11)+12.f*(-c7-c8)+16.f*(-c12)+48.f*(c0))+
                            X.z*((-c19+c20-c23+c24-c25+c26)+2.f*(-c21+c22)+4.f*(-c2+c3+c10-c11)+6.f*(c7-c8)))+
                      X.y*X2.z*((c21+c22-c23-c24+c25+c26)+2.f*(c19+c20)+3.f*(-c10-c11)+4.f*(-c1+c4+c9-c12)+6.f*(-c7-c8)+12.f*(c0))+
                      X3.z*((-c5+c6+c7-c8-c19+c20)+2.f*(c10-c11)))+
                 X3.y*(X3.z*((-c15+c16+c19-c20+c21-c22+c23-c24+c25-c26)+2.f*(c2-c3+c5-c6-c7+c8)+5.f*(-c10+c11))+
                       X2.z*((c19+c20-c23-c24+c25+c26)+2.f*(-c7-c8-c13+c14+c21+c22)+4.f*(-c5-c6+c9-c17)+6.f*(c4)+8.f*(c0-c12))+
                       X.z*((-c2+c3-c21+c22)+2.f*(c7-c8)))+
                 X3.z*(X2.y*((-c21+c22-c23+c24-c25+c26)+2.f*(c7-c8-c19+c20)+3.f*(-c2+c3+c15-c16)+4.f*(-c5+c6)+6.f*(c10-c11))+
                        X.y*((-c2+c3+c10-c11-c19+c20)+2.f*(c7-c8))));

    val +=  360.f*(X3.x*(X.y*X.z*((-c19+c20-c21+c22)+2.f*(c23-c24-c25+c26)+3.f*(c10-c11)+4.f*(-c5+c6+c7-c8))+
                      (c1+c2+c3+c4-c9-c10-c11-c12)+2.f*(-c13+c18))+
                 X2.x*(X.y*((c14-c17)+4.f*(c5+c6-c7-c8)+5.f*(-c1+c4-c9+c12))+
                       X.z*((c15-c16)+4.f*(c5-c6+c7-c8)+5.f*(-c2+c3-c10+c11)))+
                 X.x*(X3.y*X.z*((-c5+c6+c10-c11-c19+c20+c23-c24-c25+c26)+2.f*(-c21+c22)+3.f*(-c2+c3)+7.f*(c7-c8))+
                      X2.y*((c13-c18)+4.f*(c2+c3-c10-c11)+5.f*(-c1-c4+c9+c12))+
                      X.y*X3.z*((-c21+c22+c23-c24-c25+c26)+2.f*(c10-c11+c15-c16-c19+c20)+3.f*(-c2+c3-c5+c6)+5.f*(c7-c8))+
                      X2.z*((c13-c18)+4.f*(c1+c4-c9-c12)+5.f*(-c2-c3+c10+c11)))+
                 X3.y*((c1-c4+c5+c6-c7-c8+c9-c12)+2.f*(-c14+c17))+
                 X2.y*X.z*((c15-c16)+4.f*(c2-c3+c10-c11)+5.f*(-c5+c6-c7+c8))+
                 X.y*X2.z*((c14-c17)+4.f*(c1-c4+c9-c12)+5.f*(-c5-c6+c7+c8))+
                 X3.z*((c2-c3+c5-c6+c7-c8+c10-c11)+2.f*(-c15+c16)));
    
    val += 2880.f*(X.x*X.y*(c1-c4-c9+c12)+
                 X.x*X.z*(c2-c3-c10+c11)+
                 X.y*X.z*(c5-c6-c7+c8));
    
    return val*SCALE_F;
}

__inline float3 eval_g(float3 p_in, __read_only image3d_t vol)
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
    
    // Fetch coefficients
    int3 dir2x = 2*R*vecPx;
    int3 dir2y = 2*R*vecPy;
    int3 dir2z = 2*R*vecPz;

    // We re-ordered the fetches such that
    // the displacement of adjacent fetches are axis-aligned.
    // Therefore, we can move to the next coefficient with low computational overhead.
    int3 offset = org;              float c0 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 0, 0, 0)
    offset -= dir2z;                float c15 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 0,-2)
    offset += (dir2x+dir2y)>>1;     float c19 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 1,-2)
    offset += 2*dir2z;              float c20 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 1, 2)
    offset -= (dir2x+dir2y)>>1;     float c16 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 0, 2)
    offset -= (dir2y+dir2z);        float c14 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0,-2, 0)
    offset += 2*dir2y;              float c17 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 2, 0)
    offset += (dir2x-dir2z)>>1;     float c21 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 2,-1)
    offset += dir2z;                float c22 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 2, 1)
    offset -= dir2y;                float c11 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 0, 1)
    offset -= dir2x;                float c3 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //(-1, 0, 1)
    offset -= dir2z;                float c2 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //(-1, 0,-1)
    offset += dir2x;                float c10 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 0,-1)
    offset += (dir2y+dir2z)>>1;     float c12 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 1, 0)
    offset -= dir2x;                float c4 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //(-1, 1, 0)
    offset -= dir2y;                float c1 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //(-1,-1, 0)
    offset += dir2x;                float c9 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 1,-1, 0)
    offset -= (dir2x+dir2z)>>1;     float c5 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 0,-1,-1)
    offset += dir2z;                float c6 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 0,-1, 1)
    offset += (dir2y-dir2z);        float c7 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 0, 1,-1)
    offset += dir2z;                float c8 = read_imagef(vol, sp2, (int4)(offset, 1)).r;  //( 0, 1, 1)
    offset += dir2x;                float c26 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2, 1, 1)
    offset -= dir2z;                float c25 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2, 1,-1)
    offset += (dir2z-dir2y);        float c24 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2,-1, 1)
    offset -= dir2z;                float c23 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2,-1,-1)
    offset += (dir2y+dir2z)>>1;     float c18 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2, 0, 0)
    offset -= 2*dir2x;              float c13 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //(-2, 0, 0)

    // Evaluation
    
    float3 d = (float3)(0);
    float3 X = p_ref.xyz;
    float3 X2 = X*X;
    float3 X3 = X2*X;
    float3 X4 = X3*X;
    float3 X5 = X4*X;

    d.x +=    3*(((-c25-c26)+2*(c13+c15+c16-c19-c20)+3*(-c5-c6+c21+c22)+4*(-c0-c4-c10-c11-c18)+5*(c2+c3)+6*(-c1-c17)+10*(c12)+12*(c9))*X5.y+
                 ((-c2+c3-c7+c8+c10-c11+c25-c26)+2*(-c5+c6+c21-c22)+3*(c15-c16-c19+c20))*X5.z);
    d.x +=    6*((6*(-c13+c18)+37*(-c1-c2-c3-c4+c9+c10+c11+c12)));
    d.x +=   15*(((c5+c6+c7+c8+c15+c16+c21+c22)+2*(c1+c2+c3+c13-c14+c18+c23+c24)+3*(-c19-c20)+4*(c17-c25-c26)+5*(c10+c11)+6*(c12)+8*(-c0-c9)+10*(-c4))*X.x*X4.y+
                 ((-c10-c11+c15+c16+c19+c20)+2*(c13+c14+c18+c23+c24)+3*(-c5-c6-c21-c22)+4*(-c2-c3-c4-c25-c26)+5*(c7+c8)+6*(-c9)+8*(-c0+c1)+16*(c12))*X.x*X4.z+
                 ((-c15+c16)+2*(c5-c6-c23+c24)+3*(-c10+c11+c19-c20+c25-c26)+5*(c2-c3)+9*(-c7+c8))*X4.y*X.z+
                 ((-c15-c16)+2*(c5+c6+c7+c8-c13+c18+c21+c22)+3*(c10+c11)+4*(c1)+5*(-c2-c3)+6*(-c9-c17)+8*(-c12)+10*(c4))*X4.y+
                 (2*(-c4-c9+c13+c14-c23-c24)+3*(-c2-c3-c5-c6-c21-c22-c25-c26)+4*(c7+c8+c12+c17+c18)+8*(c1)+10*(c10+c11)+20*(-c0))*X.y*X4.z+
                 (2*(-c4-c7-c8+c9-c13-c14+c18+c19+c20)+3*(-c15-c16)+4*(c12)+6*(c5+c6)+7*(c2+c3-c10-c11)+8*(-c1))*X4.z);
    d.x +=   18*(((-c1-c2-c3-c4+c9+c10+c11+c12)+2*(c13-c18))*X5.x);
    d.x +=   30*(((c14-c17)+2*(c5+c6-c7-c8)+3*(-c1+c4-c9+c12))*X4.x*X.y+
                 ((c15-c16)+2*(c5-c6+c7-c8)+3*(-c2+c3-c10+c11))*X4.x*X.z+
                 ((-c14-c15-c16-c17)+4*(-c5-c6-c7-c8)+6*(c1+c2+c3+c4+c9+c10+c11+c12-c13-c18)+16*(-c0))*X4.x+
                 ((c5+c6)+2*(-c2-c3)+4*(c13+c14)+7*(c15+c16)+8*(-c18-c25-c26)+10*(c17+c23+c24)+11*(-c19-c20-c21-c22)+13*(c7+c8+c10+c11)+16*(c1)+20*(-c4)+38*(-c9)+56*(-c0)+64*(c12))*X.x*X2.y*X2.z+
                 ((c19+c20+c21+c22+c23+c24+c25+c26)+4*(c15+c16)+5*(-c5-c6-c7-c8-c10-c11)+6*(c12-c13-c14-c17-c18)+8*(-c2-c3)+12*(c9)+18*(c1+c4))*X.x*X2.y+
                 ((c19+c20+c21+c22+c23+c24+c25+c26)+2*(-c9)+4*(c14+c17)+5*(-c5-c6-c7-c8)+6*(-c13-c15-c16-c18)+8*(-c1-c4-c12)+9*(c10+c11)+18*(c2+c3))*X.x*X2.z+
                 ((-c14-c15-c16-c17)+6*(c13+c18)+8*(-c5-c6-c7-c8)+10*(c1+c2+c3+c4+c9+c10+c11+c12)+56*(-c0))*X.x+
                 ((c2+c3-c5-c6)+2*(c13-c21-c22)+3*(c15+c16-c19-c20-c23-c24)+4*(c1+c7+c8+c9+c17-c25-c26)+5*(c10+c11)+6*(-c4)+8*(c18)+12*(c12)+28*(-c0))*X3.y*X2.z+
                 ((-c5-c6-c7-c8+c10+c11+c21+c22)+4*(c0-c12))*X3.y+
                 (2*(-c15+c16+c19-c20)+3*(c21-c22-c23+c24)+4*(-c10+c11+c25-c26)+5*(c2-c3)+6*(c5-c6)+11*(-c7+c8))*X2.y*X3.z+
                 ((c2+c3)+3*(-c15-c16)+4*(-c9-c13+c23+c24+c25+c26)+5*(c19+c20+c21+c22)+6*(-c17)+8*(-c1-c18)+10*(c4-c10-c11)+12*(-c7-c8)+16*(-c12)+48*(c0))*X2.y*X2.z+
                 ((-c19+c20-c23+c24-c25+c26)+2*(-c21+c22)+4*(-c2+c3+c10-c11)+6*(c7-c8))*X2.y*X.z+
                 ((c21+c22-c23-c24+c25+c26)+2*(c19+c20)+3*(-c10-c11)+4*(-c1+c4+c9-c12)+6*(-c7-c8)+12*(c0))*X.y*X2.z+
                 ((-c5+c6+c7-c8-c19+c20)+2*(c10-c11))*X3.z);
    d.x +=   60*(((c1+c4+c5+c6+c7+c8-c13+c14+c15+c16+c17+c23+c24+c25+c26)+2*(c10+c11-c19-c20-c21-c22)+3*(-c18)+5*(-c9)+7*(c12)+8*(-c0))*X3.x*X2.y+
                 ((c2+c3+c5+c6+c7+c8+c10+c11-c13+c14+c15+c16+c17+c23+c24+c25+c26)+2*(-c19-c20-c21-c22)+3*(-c18)+4*(-c9)+8*(-c0+c12))*X3.x*X2.z+
                 ((c14+c15+c16+c17+c23+c24+c25+c26)+5*(c5+c6+c7+c8)+6*(c13+c18)+7*(-c1-c2-c3-c4)+9*(-c9-c10-c11-c12)+24*(c0))*X3.x+
                 (2*(c21-c22)+3*(c5-c6+c23-c24+c25-c26)+4*(c19-c20)+6*(c2-c3)+9*(-c7+c8)+16*(-c10+c11))*X.x*X3.y*X.z+
                 ((-c5-c6-c7-c8-c15-c16+c19+c20-c23-c24+c25+c26)+2*(c9+c14+c21+c22)+4*(-c1+c4-c17)+8*(c0-c12))*X.x*X3.y+
                 ((-c15+c16)+4*(-c19+c20)+5*(-c5+c6-c21+c22-c23+c24-c25+c26)+8*(-c2+c3)+13*(c7-c8)+25*(c10-c11))*X.x*X2.y*X.z+
                 (2*(-c15+c16+c19-c20)+3*(c23-c24+c25-c26)+4*(c21-c22)+5*(c5-c6)+6*(c2-c3)+7*(-c7+c8)+14*(-c10+c11))*X.x*X.y*X3.z+
                 ((c5+c6)+2*(-c17)+3*(-c15-c16)+4*(-c14+c21+c22)+5*(c19+c20-c23-c24+c25+c26)+6*(-c10-c11)+7*(-c7-c8)+8*(-c1+c4)+22*(c9)+24*(c0)+28*(-c12))*X.x*X.y*X2.z+
                 ((-c19+c20-c21+c22)+2*(c23-c24-c25+c26)+3*(c10-c11)+4*(-c5+c6+c7-c8))*X.x*X.y*X.z+
                 ((-c21+c22-c23+c24-c25+c26)+2*(-c19+c20)+3*(-c5+c6+c7-c8+c15-c16)+4*(-c2+c3)+5*(c10-c11))*X.x*X3.z+
                 ((-c5+c6+c10-c11-c19+c20+c23-c24-c25+c26)+2*(-c21+c22)+3*(-c2+c3)+7*(c7-c8))*X3.y*X.z+
                 ((c13-c18)+4*(c2+c3-c10-c11)+5*(-c1-c4+c9+c12))*X2.y+
                 ((-c21+c22+c23-c24-c25+c26)+2*(c10-c11+c15-c16-c19+c20)+3*(-c2+c3-c5+c6)+5*(c7-c8))*X.y*X3.z+
                 ((c13-c18)+4*(c1+c4-c9-c12)+5*(-c2-c3+c10+c11))*X2.z);
    d.x +=   90*(((c10+c11+c15+c16-c19-c20-c21-c22-c23-c24-c25-c26)+2*(c1-c4+c7+c8+c12+c17)+4*(c18)+8*(-c0))*X2.x*X3.y+
                 ((-c15+c16)+3*(c19-c20+c21-c22-c23+c24+c25-c26)+4*(c2-c3)+5*(-c10+c11)+6*(c5-c6)+12*(-c7+c8))*X2.x*X2.y*X.z+
                 ((-c10-c11+c19+c20+c21+c22)+2*(-c9+c13-c15-c16+c23+c24+c25+c26)+4*(-c1-c4)+6*(-c18)+8*(-c12)+16*(c0))*X2.x*X2.y+
                 (2*(-c9+c14)+3*(c10+c11+c15+c16-c19-c20-c21-c22-c23-c24-c25-c26)+4*(c1-c4+c17)+6*(c7+c8)+8*(c12)+12*(c18)+24*(-c0))*X2.x*X.y*X2.z+
                 ((-c5-c6+c7+c8-c23-c24+c25+c26)+4*(c9-c12))*X2.x*X.y+
                 ((-c10+c11-c15+c16+c19-c20+c21-c22-c23+c24+c25-c26)+2*(c2-c3+c5-c6)+4*(-c7+c8))*X2.x*X3.z+
                 ((c19+c20+c21+c22)+2*(c9+c13-c14-c17+c23+c24+c25+c26)+4*(-c2-c3-c12)+5*(-c10-c11)+6*(-c18)+16*(c0))*X2.x*X2.z+
                 ((-c5+c6-c7+c8-c23+c24-c25+c26)+4*(c10-c11))*X2.x*X.z);
    d.x +=  120*(((c23-c24-c25+c26)+2*(c19-c20+c21-c22)+3*(c5-c6-c7+c8)+6*(-c10+c11))*X3.x*X.y*X.z+
                 ((-c14+c17-c23-c24+c25+c26)+2*(c1-c4-c5-c6+c7+c8)+6*(c9-c12))*X3.x*X.y+
                 ((-c15+c16-c23+c24-c25+c26)+2*(c2-c3-c5+c6-c7+c8)+6*(c10-c11))*X3.x*X.z+
                 ((c14-c17)+4*(c5+c6-c7-c8)+5*(-c1+c4-c9+c12))*X.x*X.y+
                 ((c15-c16)+4*(c5-c6+c7-c8)+5*(-c2+c3-c10+c11))*X.x*X.z);
    d.x +=  180*(((-c19+c20-c21+c22)+2*(c23-c24-c25+c26)+3*(c10-c11)+4*(-c5+c6+c7-c8))*X2.x*X.y*X.z+
                 ((c1+c2+c3+c4-c9-c10-c11-c12)+2*(-c13+c18))*X2.x);
    d.x +=  480*(((c1-c4-c9+c12))*X.y+
                 ((c2-c3-c10+c11))*X.z);

    d.y +=    3*(((-c5+c6+c7-c8-c10+c11+c21-c22)+2*(-c2+c3+c25-c26)+3*(c15-c16-c19+c20))*X5.z);
    d.y +=    6*(((c14-c17)+2*(c5+c6-c7-c8)+3*(-c1+c4-c9+c12))*X5.x+
                 (6*(-c14+c17)+37*(-c1+c4-c5-c6+c7+c8-c9+c12)));
    d.y +=    9*(((-c2-c3-c15-c16+c19+c20+c25+c26)+2*(-c1+c9-c18-c21-c22)+4*(c4+c14)+5*(-c10-c11)+7*(c7+c8)+8*(-c0-c17)+10*(c12))*X5.y);
    d.y +=   15*(((-c25-c26)+2*(c13+c15+c16-c19-c20)+3*(-c5-c6+c21+c22)+4*(-c0-c4-c10-c11-c18)+5*(c2+c3)+6*(-c1-c17)+10*(c12)+12*(c9))*X.x*X4.y+
                 (2*(-c4-c9+c13+c14-c23-c24)+3*(-c2-c3-c5-c6-c21-c22-c25-c26)+4*(c7+c8+c12+c17+c18)+8*(c1)+10*(c10+c11)+20*(-c0))*X.x*X4.z+
                 ((c19-c20)+2*(c25-c26)+3*(c10-c11+c15-c16-c21+c22)+6*(c2-c3)+7*(-c7+c8)+9*(-c5+c6))*X4.y*X.z+
                 ((-c15-c16)+2*(-c10-c11-c13)+3*(c7+c8)+6*(-c2-c3+c4+c9)+9*(c5+c6)+12*(c1-c14)+16*(-c0))*X4.y+
                 ((-c10-c11+c15+c16+c19+c20)+2*(c13+c14+c17+c23+c24)+3*(-c2-c3-c25-c26)+4*(-c5-c6-c9-c21-c22)+5*(c7+c8)+6*(-c4)+8*(-c0+c1)+16*(c12))*X.y*X4.z+
                 (2*(c4-c9-c10-c11-c13-c14+c17+c19+c20)+3*(-c15-c16)+4*(c12)+6*(c2+c3)+7*(c5+c6-c7-c8)+8*(-c1))*X4.z);
    d.y +=   30*(((c1+c4+c5+c6+c7+c8-c13+c14+c15+c16+c17+c23+c24+c25+c26)+2*(c10+c11-c19-c20-c21-c22)+3*(-c18)+5*(-c9)+7*(c12)+8*(-c0))*X4.x*X.y+
                 ((c23-c24-c25+c26)+2*(c19-c20+c21-c22)+3*(c5-c6-c7+c8)+6*(-c10+c11))*X4.x*X.z+
                 ((-c14+c17-c23-c24+c25+c26)+2*(c1-c4-c5-c6+c7+c8)+6*(c9-c12))*X4.x+
                 (2*(-c9+c14)+3*(c10+c11+c15+c16-c19-c20-c21-c22-c23-c24-c25-c26)+4*(c1-c4+c17)+6*(c7+c8)+8*(c12)+12*(c18)+24*(-c0))*X3.x*X2.z+
                 ((-c5-c6+c7+c8-c23-c24+c25+c26)+4*(c9-c12))*X3.x+
                 ((c5+c6+c7+c8+c15+c16+c21+c22)+2*(c1+c2+c3+c13-c14+c18+c23+c24)+3*(-c19-c20)+4*(c17-c25-c26)+5*(c10+c11)+6*(c12)+8*(-c0-c9)+10*(-c4))*X2.x*X3.y+
                 ((c5+c6)+2*(-c2-c3)+4*(c13+c14)+7*(c15+c16)+8*(-c18-c25-c26)+10*(c17+c23+c24)+11*(-c19-c20-c21-c22)+13*(c7+c8+c10+c11)+16*(c1)+20*(-c4)+38*(-c9)+56*(-c0)+64*(c12))*X2.x*X.y*X2.z+
                 ((c19+c20+c21+c22+c23+c24+c25+c26)+4*(c15+c16)+5*(-c5-c6-c7-c8-c10-c11)+6*(c12-c13-c14-c17-c18)+8*(-c2-c3)+12*(c9)+18*(c1+c4))*X2.x*X.y+
                 (2*(-c15+c16+c19-c20)+3*(c23-c24+c25-c26)+4*(c21-c22)+5*(c5-c6)+6*(c2-c3)+7*(-c7+c8)+14*(-c10+c11))*X2.x*X3.z+
                 ((c5+c6)+2*(-c17)+3*(-c15-c16)+4*(-c14+c21+c22)+5*(c19+c20-c23-c24+c25+c26)+6*(-c10-c11)+7*(-c7-c8)+8*(-c1+c4)+22*(c9)+24*(c0)+28*(-c12))*X2.x*X2.z+
                 ((-c19+c20-c21+c22)+2*(c23-c24-c25+c26)+3*(c10-c11)+4*(-c5+c6+c7-c8))*X2.x*X.z+
                 ((c21+c22-c23-c24+c25+c26)+2*(c19+c20)+3*(-c10-c11)+4*(-c1+c4+c9-c12)+6*(-c7-c8)+12*(c0))*X.x*X2.z+
                 ((c2+c3+c21+c22)+2*(-c7-c8+c13-c14+c15+c16+c23+c24)+3*(c5+c6-c25-c26)+4*(c17-c19-c20)+6*(-c4+c10+c11)+8*(-c0+c12)+10*(-c9))*X3.y*X2.z+
                 (2*(c13+c15+c16+c18+c21+c22)+9*(c2+c3+c10+c11)+12*(-c4-c9+c14+c17)+13*(-c5-c6)+14*(-c1)+15*(-c7-c8)+18*(-c12)+40*(c0))*X3.y+
                 ((c19+c20+c21+c22+c23+c24+c25+c26)+2*(-c4)+4*(c13+c18)+5*(-c2-c3)+6*(-c14-c15-c16-c17)+8*(-c1-c9-c10-c11-c12)+12*(c7+c8)+18*(c5+c6))*X.y*X2.z+
                 ((-c13-c15-c16-c18)+6*(c14+c17)+8*(-c2-c3-c10-c11)+10*(c1+c4+c5+c6+c7+c8+c9+c12)+56*(-c0))*X.y+
                 ((-c2+c3+c10-c11-c19+c20)+2*(c7-c8))*X3.z);
    d.y +=   60*(((-c15+c16)+3*(c19-c20+c21-c22-c23+c24+c25-c26)+4*(c2-c3)+5*(-c10+c11)+6*(c5-c6)+12*(-c7+c8))*X3.x*X.y*X.z+
                 ((-c10-c11+c19+c20+c21+c22)+2*(-c9+c13-c15-c16+c23+c24+c25+c26)+4*(-c1-c4)+6*(-c18)+8*(-c12)+16*(c0))*X3.x*X.y+
                 ((-c19+c20-c21+c22)+2*(c23-c24-c25+c26)+3*(c10-c11)+4*(-c5+c6+c7-c8))*X3.x*X.z+
                 ((-c15+c16)+4*(-c19+c20)+5*(-c5+c6-c21+c22-c23+c24-c25+c26)+8*(-c2+c3)+13*(c7-c8)+25*(c10-c11))*X2.x*X.y*X.z+
                 ((c14-c17)+4*(c5+c6-c7-c8)+5*(-c1+c4-c9+c12))*X2.x+
                 ((-c15+c16)+2*(c5-c6-c23+c24)+3*(-c10+c11+c19-c20+c25-c26)+5*(c2-c3)+9*(-c7+c8))*X.x*X3.y*X.z+
                 ((-c15-c16)+2*(c5+c6+c7+c8-c13+c18+c21+c22)+3*(c10+c11)+4*(c1)+5*(-c2-c3)+6*(-c9-c17)+8*(-c12)+10*(c4))*X.x*X3.y+
                 (2*(-c15+c16+c19-c20)+3*(c21-c22-c23+c24)+4*(-c10+c11+c25-c26)+5*(c2-c3)+6*(c5-c6)+11*(-c7+c8))*X.x*X.y*X3.z+
                 ((c2+c3)+3*(-c15-c16)+4*(-c9-c13+c23+c24+c25+c26)+5*(c19+c20+c21+c22)+6*(-c17)+8*(-c1-c18)+10*(c4-c10-c11)+12*(-c7-c8)+16*(-c12)+48*(c0))*X.x*X.y*X2.z+
                 ((-c19+c20-c23+c24-c25+c26)+2*(-c21+c22)+4*(-c2+c3+c10-c11)+6*(c7-c8))*X.x*X.y*X.z+
                 ((-c21+c22+c23-c24-c25+c26)+2*(c10-c11+c15-c16-c19+c20)+3*(-c2+c3-c5+c6)+5*(c7-c8))*X.x*X3.z+
                 (2*(-c15+c16-c21+c22)+3*(-c10+c11)+5*(-c2+c3+c5-c6)+9*(c7-c8))*X3.y*X.z+
                 ((-c21+c22-c23+c24-c25+c26)+2*(c7-c8-c19+c20)+3*(-c2+c3+c15-c16)+4*(-c5+c6)+6*(c10-c11))*X.y*X3.z+
                 ((c14-c17)+4*(c1-c4+c9-c12)+5*(-c5-c6+c7+c8))*X2.z);
    d.y +=   90*(((c10+c11+c15+c16-c19-c20-c21-c22-c23-c24-c25-c26)+2*(c1-c4+c7+c8+c12+c17)+4*(c18)+8*(-c0))*X3.x*X2.y+
                 (2*(c21-c22)+3*(c5-c6+c23-c24+c25-c26)+4*(c19-c20)+6*(c2-c3)+9*(-c7+c8)+16*(-c10+c11))*X2.x*X2.y*X.z+
                 ((-c5-c6-c7-c8-c15-c16+c19+c20-c23-c24+c25+c26)+2*(c9+c14+c21+c22)+4*(-c1+c4-c17)+8*(c0-c12))*X2.x*X2.y+
                 ((c2+c3-c5-c6)+2*(c13-c21-c22)+3*(c15+c16-c19-c20-c23-c24)+4*(c1+c7+c8+c9+c17-c25-c26)+5*(c10+c11)+6*(-c4)+8*(c18)+12*(c12)+28*(-c0))*X.x*X2.y*X2.z+
                 ((-c5-c6-c7-c8+c10+c11+c21+c22)+4*(c0-c12))*X.x*X2.y+
                 ((-c15+c16+c19-c20+c21-c22+c23-c24+c25-c26)+2*(c2-c3+c5-c6-c7+c8)+5*(-c10+c11))*X2.y*X3.z+
                 ((c19+c20-c23-c24+c25+c26)+2*(-c7-c8-c13+c14+c21+c22)+4*(-c5-c6+c9-c17)+6*(c4)+8*(c0-c12))*X2.y*X2.z+
                 ((-c2+c3-c21+c22)+2*(c7-c8))*X2.y*X.z);
    d.y +=  120*(((c13-c18)+4*(c2+c3-c10-c11)+5*(-c1-c4+c9+c12))*X.x*X.y+
                 ((c15-c16)+4*(c2-c3+c10-c11)+5*(-c5+c6-c7+c8))*X.y*X.z);
    d.y +=  180*(((-c5+c6+c10-c11-c19+c20+c23-c24-c25+c26)+2*(-c21+c22)+3*(-c2+c3)+7*(c7-c8))*X.x*X2.y*X.z+
                 ((c1-c4+c5+c6-c7-c8+c9-c12)+2*(-c14+c17))*X2.y);
    d.y +=  480*(((c1-c4-c9+c12))*X.x+
                 ((c5-c6-c7+c8))*X.z);

    d.z +=    3*(((c19-c20)+2*(c25-c26)+3*(c10-c11+c15-c16-c21+c22)+6*(c2-c3)+7*(-c7+c8)+9*(-c5+c6))*X5.y);
    d.z +=    6*(((c15-c16)+2*(c5-c6+c7-c8)+3*(-c2+c3-c10+c11))*X5.x+
                 (6*(-c15+c16)+37*(-c2+c3-c5+c6-c7+c8-c10+c11)));
    d.z +=    9*(((c2+c3+c5+c6+c21+c22+c25+c26)+2*(-c4-c9-c15-c16-c17-c18-c19-c20)+6*(c7+c8+c10+c11)+8*(-c0-c12))*X5.z);
    d.z +=   15*(((-c15+c16)+2*(c5-c6-c23+c24)+3*(-c10+c11+c19-c20+c25-c26)+5*(c2-c3)+9*(-c7+c8))*X.x*X4.y+
                 ((-c2+c3-c7+c8+c10-c11+c25-c26)+2*(-c5+c6+c21-c22)+3*(c15-c16-c19+c20))*X.x*X4.z+
                 ((c2+c3+c21+c22)+2*(-c7-c8+c13-c14+c15+c16+c23+c24)+3*(c5+c6-c25-c26)+4*(c17-c19-c20)+6*(-c4+c10+c11)+8*(-c0+c12)+10*(-c9))*X4.y*X.z+
                 (2*(-c15+c16-c21+c22)+3*(-c10+c11)+5*(-c2+c3+c5-c6)+9*(c7-c8))*X4.y+
                 ((-c5+c6+c7-c8-c10+c11+c21-c22)+2*(-c2+c3+c25-c26)+3*(c15-c16-c19+c20))*X.y*X4.z);
    d.z +=   30*(((c23-c24-c25+c26)+2*(c19-c20+c21-c22)+3*(c5-c6-c7+c8)+6*(-c10+c11))*X4.x*X.y+
                 ((c2+c3+c5+c6+c7+c8+c10+c11-c13+c14+c15+c16+c17+c23+c24+c25+c26)+2*(-c19-c20-c21-c22)+3*(-c18)+4*(-c9)+8*(-c0+c12))*X4.x*X.z+
                 ((-c15+c16-c23+c24-c25+c26)+2*(c2-c3-c5+c6-c7+c8)+6*(c10-c11))*X4.x+
                 ((-c15+c16)+3*(c19-c20+c21-c22-c23+c24+c25-c26)+4*(c2-c3)+5*(-c10+c11)+6*(c5-c6)+12*(-c7+c8))*X3.x*X2.y+
                 ((-c5+c6-c7+c8-c23+c24-c25+c26)+4*(c10-c11))*X3.x+
                 (2*(c21-c22)+3*(c5-c6+c23-c24+c25-c26)+4*(c19-c20)+6*(c2-c3)+9*(-c7+c8)+16*(-c10+c11))*X2.x*X3.y+
                 ((c5+c6)+2*(-c2-c3)+4*(c13+c14)+7*(c15+c16)+8*(-c18-c25-c26)+10*(c17+c23+c24)+11*(-c19-c20-c21-c22)+13*(c7+c8+c10+c11)+16*(c1)+20*(-c4)+38*(-c9)+56*(-c0)+64*(c12))*X2.x*X2.y*X.z+
                 ((-c15+c16)+4*(-c19+c20)+5*(-c5+c6-c21+c22-c23+c24-c25+c26)+8*(-c2+c3)+13*(c7-c8)+25*(c10-c11))*X2.x*X2.y+
                 ((-c19+c20-c21+c22)+2*(c23-c24-c25+c26)+3*(c10-c11)+4*(-c5+c6+c7-c8))*X2.x*X.y+
                 ((-c10-c11+c15+c16+c19+c20)+2*(c13+c14+c18+c23+c24)+3*(-c5-c6-c21-c22)+4*(-c2-c3-c4-c25-c26)+5*(c7+c8)+6*(-c9)+8*(-c0+c1)+16*(c12))*X2.x*X3.z+
                 ((c19+c20+c21+c22+c23+c24+c25+c26)+2*(-c9)+4*(c14+c17)+5*(-c5-c6-c7-c8)+6*(-c13-c15-c16-c18)+8*(-c1-c4-c12)+9*(c10+c11)+18*(c2+c3))*X2.x*X.z+
                 ((-c19+c20-c23+c24-c25+c26)+2*(-c21+c22)+4*(-c2+c3+c10-c11)+6*(c7-c8))*X.x*X2.y+
                 ((-c2+c3-c21+c22)+2*(c7-c8))*X3.y+
                 ((-c10-c11+c15+c16+c19+c20)+2*(c13+c14+c17+c23+c24)+3*(-c2-c3-c25-c26)+4*(-c5-c6-c9-c21-c22)+5*(c7+c8)+6*(-c4)+8*(-c0+c1)+16*(c12))*X2.y*X3.z+
                 ((c19+c20+c21+c22+c23+c24+c25+c26)+2*(-c4)+4*(c13+c18)+5*(-c2-c3)+6*(-c14-c15-c16-c17)+8*(-c1-c9-c10-c11-c12)+12*(c7+c8)+18*(c5+c6))*X2.y*X.z+
                 (2*(c13+c14+c17+c18+c19+c20)+8*(c1+c12)+10*(c4+c9)+12*(c15+c16)+13*(-c2-c3-c5-c6)+15*(-c7-c8-c10-c11)+40*(c0))*X3.z+
                 ((-c13-c14-c17-c18)+6*(c15+c16)+8*(-c1-c4-c9-c12)+10*(c2+c3+c5+c6+c7+c8+c10+c11)+56*(-c0))*X.z);
    d.z +=   45*(((c2-c3+c5-c6+c7-c8+c10-c11)+2*(-c15+c16))*X4.z);
    d.z +=   60*((2*(-c9+c14)+3*(c10+c11+c15+c16-c19-c20-c21-c22-c23-c24-c25-c26)+4*(c1-c4+c17)+6*(c7+c8)+8*(c12)+12*(c18)+24*(-c0))*X3.x*X.y*X.z+
                 ((-c19+c20-c21+c22)+2*(c23-c24-c25+c26)+3*(c10-c11)+4*(-c5+c6+c7-c8))*X3.x*X.y+
                 ((c19+c20+c21+c22)+2*(c9+c13-c14-c17+c23+c24+c25+c26)+4*(-c2-c3-c12)+5*(-c10-c11)+6*(-c18)+16*(c0))*X3.x*X.z+
                 ((c5+c6)+2*(-c17)+3*(-c15-c16)+4*(-c14+c21+c22)+5*(c19+c20-c23-c24+c25+c26)+6*(-c10-c11)+7*(-c7-c8)+8*(-c1+c4)+22*(c9)+24*(c0)+28*(-c12))*X2.x*X.y*X.z+
                 ((c15-c16)+4*(c5-c6+c7-c8)+5*(-c2+c3-c10+c11))*X2.x+
                 ((c2+c3-c5-c6)+2*(c13-c21-c22)+3*(c15+c16-c19-c20-c23-c24)+4*(c1+c7+c8+c9+c17-c25-c26)+5*(c10+c11)+6*(-c4)+8*(c18)+12*(c12)+28*(-c0))*X.x*X3.y*X.z+
                 ((-c5+c6+c10-c11-c19+c20+c23-c24-c25+c26)+2*(-c21+c22)+3*(-c2+c3)+7*(c7-c8))*X.x*X3.y+
                 ((c2+c3)+3*(-c15-c16)+4*(-c9-c13+c23+c24+c25+c26)+5*(c19+c20+c21+c22)+6*(-c17)+8*(-c1-c18)+10*(c4-c10-c11)+12*(-c7-c8)+16*(-c12)+48*(c0))*X.x*X2.y*X.z+
                 (2*(-c4-c9+c13+c14-c23-c24)+3*(-c2-c3-c5-c6-c21-c22-c25-c26)+4*(c7+c8+c12+c17+c18)+8*(c1)+10*(c10+c11)+20*(-c0))*X.x*X.y*X3.z+
                 ((c21+c22-c23-c24+c25+c26)+2*(c19+c20)+3*(-c10-c11)+4*(-c1+c4+c9-c12)+6*(-c7-c8)+12*(c0))*X.x*X.y*X.z+
                 (2*(-c4-c7-c8+c9-c13-c14+c18+c19+c20)+3*(-c15-c16)+4*(c12)+6*(c5+c6)+7*(c2+c3-c10-c11)+8*(-c1))*X.x*X3.z+
                 ((c19+c20-c23-c24+c25+c26)+2*(-c7-c8-c13+c14+c21+c22)+4*(-c5-c6+c9-c17)+6*(c4)+8*(c0-c12))*X3.y*X.z+
                 ((c15-c16)+4*(c2-c3+c10-c11)+5*(-c5+c6-c7+c8))*X2.y+
                 (2*(c4-c9-c10-c11-c13-c14+c17+c19+c20)+3*(-c15-c16)+4*(c12)+6*(c2+c3)+7*(c5+c6-c7-c8)+8*(-c1))*X.y*X3.z);
    d.z +=   90*(((-c10+c11-c15+c16+c19-c20+c21-c22-c23+c24+c25-c26)+2*(c2-c3+c5-c6)+4*(-c7+c8))*X3.x*X2.z+
                 (2*(-c15+c16+c19-c20)+3*(c23-c24+c25-c26)+4*(c21-c22)+5*(c5-c6)+6*(c2-c3)+7*(-c7+c8)+14*(-c10+c11))*X2.x*X.y*X2.z+
                 ((-c21+c22-c23+c24-c25+c26)+2*(-c19+c20)+3*(-c5+c6+c7-c8+c15-c16)+4*(-c2+c3)+5*(c10-c11))*X2.x*X2.z+
                 (2*(-c15+c16+c19-c20)+3*(c21-c22-c23+c24)+4*(-c10+c11+c25-c26)+5*(c2-c3)+6*(c5-c6)+11*(-c7+c8))*X.x*X2.y*X2.z+
                 ((-c5+c6+c7-c8-c19+c20)+2*(c10-c11))*X.x*X2.z+
                 ((-c15+c16+c19-c20+c21-c22+c23-c24+c25-c26)+2*(c2-c3+c5-c6-c7+c8)+5*(-c10+c11))*X3.y*X2.z+
                 ((-c21+c22-c23+c24-c25+c26)+2*(c7-c8-c19+c20)+3*(-c2+c3+c15-c16)+4*(-c5+c6)+6*(c10-c11))*X2.y*X2.z+
                 ((-c2+c3+c10-c11-c19+c20)+2*(c7-c8))*X.y*X2.z);
    d.z +=  120*(((c13-c18)+4*(c1+c4-c9-c12)+5*(-c2-c3+c10+c11))*X.x*X.z+
                 ((c14-c17)+4*(c1-c4+c9-c12)+5*(-c5-c6+c7+c8))*X.y*X.z);
    d.z +=  180*(((-c21+c22+c23-c24-c25+c26)+2*(c10-c11+c15-c16-c19+c20)+3*(-c2+c3-c5+c6)+5*(c7-c8))*X.x*X.y*X2.z+
                 ((c2-c3+c5-c6+c7-c8+c10-c11)+2*(-c15+c16))*X2.z);
    d.z +=  480*(((c2-c3-c10+c11))*X.x+
                 ((c5-c6-c7+c8))*X.y);

    // [dx dy dz]^T_{\pi_n} = R_n*P_n*[dx dy dz]^T_{\pi_0}
    d = (float3)(dot(convert_float3((int3)(vecPx.x, vecPy.x, vecPz.x)), d), 
                 dot(convert_float3((int3)(vecPx.y, vecPy.y, vecPz.y)), d), 
                 dot(convert_float3((int3)(vecPx.z, vecPy.z, vecPz.z)), d));
    d *= convert_float3(R);


    return d*SCALE_G;
}

