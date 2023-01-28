#define DENOM_M 0.00260417
#define DENOM_G 0.015625f  // 1/64

#define TYPE_BLUE   0
#define TYPE_GREEN  1
#define TYPE_RED    2

#define c0  c[0]
#define c1  c[1]
#define c2  c[2]
#define c3  c[3]
#define c4  c[4]
#define c5  c[5]
#define c6  c[6]
#define c7  c[7]
#define c8  c[8]
#define c9  c[9]
#define c10 c[10]
#define c11 c[11]
#define c12 c[12]
#define c13 c[13]
#define c14 c[14]
#define c15 c[15]
#define c16 c[16]
#define c17 c[17]
#define c18 c[18]
#define c19 c[19]
#define c20 c[20]
#define c21 c[21]
#define c22 c[22]
#define c23 c[23]
#define c24 c[24]
#define c25 c[25]
#define c26 c[26]
#define c27 c[27]
#define c28 c[28]
#define c29 c[29]
#define c30 c[30]
#define c31 c[31]
#define c32 c[32]
#define c33 c[33]
#define c34 c[34]
#define c35 c[35]
#define c36 c[36]
#define c37 c[37]

__inline float eval_M_expr_rgb(float* c, float4 u1, float4 u2, float4 u3)
{
    return  4*(((c2+c3+c9+c12+c22+c25+c34+c35) + 4*(c7+c8+c14+c15+c20+c21+c27+c28) + 14*(c10+c11+c23+c24))*u3.y +
               ((c2+c4+c7+c17+c20+c30+c34+c36) + 4*(c9+c11+c13+c15+c22+c24+c26+c28) + 14*(c10+c14+c23+c27))*u3.z) +
            12*(((c2+c3+c8+c21+c34+c35) + 2*(c9+c22) + 3*(c7+c20) + 5*(c15+c28) + 7*(c14+c27) + 11*(c11+c24) + 17*(c10+c23))*u1.y +
                ((c2+c4+c13+c26+c34+c36) + 2*(c7+c20) + 3*(c9+c22) + 5*(c15+c28) + 7*(c11+c24) + 11*(c14+c27) + 17*(c10+c23))*u1.z)*u1.y*u1.z;
}

__inline float eval_M_expr_rg(float* c, float4 u1, float4 u2, float4 u3)
{
    return (( 8*((c0+c1+c3+c4+c6+c8+c13+c15+c20+c22+c24+c27) + 4*(c2+c7+c9+c11+c14+c23) + 12*c10)*u1.x +
             12*(((c0+c4+c6+c13) + 2*(c3+c22) + 3*(c8+c15+c20+c27) + 4*(c2+c9+c24) + 8*(c7+c14) + 12*(c11+c23) + 24*c10)*u1.y +
                 ((c1+c3+c6+c8) + 2*(c4+c20) + 3*(c13+c15+c22+c24) + 4*(c2+c7+c27) + 8*(c9+c11) + 12*(c14+c23) + 24*c10)*u1.z))*u1.x +
             24*(((c2+c3+c9+c21+c22+c28) + 2*(c8+c15+c20+c27) + 3*(c7+c14) + 4*c24 + 7*(c11+c23) + 10*c10)*u2.y +
                 ((c3+c4+c8+c13) + 2*(c2+c28) + 3*(c20+c22) + 4*(c7+c9+c15) + 6*(c24+c27) + 10*(c11+c14) + 16*c23 + 22*c10)*u1.y*u1.z +
                 ((c2+c4+c7+c20+c26+c28) + 2*(c13+c15+c22+c24) + 3*(c9+c11) + 4*c27 + 7*(c14+c23) + 10*c10)*u2.z))*u1.x;
}

__inline float eval_M_expr_gb(float* c, float4 u1, float4 u2, float4 u3)
{
    return ( 4*((c7+c8+c9+c12+c13+c16+c17+c18) + 4*(c2+c3+c4+c5+c23+c24+c27+c28) + 14*(c10+c11+c14+c15))*u2.w +
            12*(((c9+c12+c20+c21+c22+c25) + 2*(c2+c3) + 3*(c7+c8) + 5*(c27+c28) + 7*(c14+c15) + 11*(c23+c24) + 17*(c10+c11))*u2.y +
                ((c4+c5+c9+c12+c13+c16) + 2*(c7+c8) + 3*(c2+c3) + 5*(c27+c28) + 7*(c23+c24) + 11*(c14+c15) + 17*(c10+c11))*u1.y*u1.w +
                ((c7+c17+c20+c22+c26+c30) + 2*(c2+c4) + 3*(c9+c13) + 5*(c24+c28) + 7*(c11+c15) + 11*(c23+c27) + 17*(c10+c14))*u2.z +
                ((c3+c5+c7+c8+c17+c18) + 2*(c9+c13) + 3*(c2+c4) + 5*(c24+c28) + 7*(c23+c27) + 11*(c11+c15) + 17*(c10+c14))*u1.z*u1.w) +
            24*((c3+c4+c8+c13+c20+c22) + 2*(c2+c7+c9) + 6*(c28) + 8*(c15+c24+c27) + 12*(c11+c14+c23) + 18*(c10))*u1.y*u1.z)*u1.w;
            
}

__inline float eval_M_expr_red(float* c, float4 u1, float4 u2, float4 u3)
{
    return (    ((c0+c1+c3+c4+c32+c33+c35+c36) + 4*(c2+c6+c8+c13+c15+c19+c21+c26+c28+c34) + 23*(c7+c9+c11+c14+c20+c22+c24+c27) + 76*(c10+c23))*u2.w + 
             3*(((c0+c4+c6+c13+c19+c26+c32+c36) + 2*(c3+c35) + 4*(c2+c34) + 7*(c8+c15+c21+c28) + 14*(c9+c22) + 23*(c7+c14+c20+c27) + 32*(c11+c24) + 76*(c10+c23))*u1.y*u1.w +
                ((c1+c3+c6+c8+c19+c21+c33+c35) + 2*(c4+c36) + 4*(c2+c34) + 7*(c13+c15+c26+c28) + 14*(c7+c20) + 23*(c9+c11+c22+c24) + 32*(c14+c27) + 76*(c10+c23))*u1.z*u1.w) +
             6*((c0+c1+c3+c4+c19+c21+c26+c28) + 3*(c6+c8+c13+c15) + 4*(c2) + 9*(c20+c22+c24+c27) + 14*(c7+c9+c11+c14) + 32*(c23) + 44*(c10))*u1.x*u1.w +
             12*(((c0+c1+c3+c4) + 2*(c6+c8+c13+c15) + 3*(c20+c22+c24+c27) + 4*(c2) + 8*(c7+c9+c11+c14) + 12*(c23) + 24*(c10))*u2.x +
                 ((c0+c4+c6+c13) + 2*(c3+c21+c28) + 4*(c2) + 5*(c8+c15) + 6*(c22) + 8*(c9) + 9*(c20+c27) + 12*(c24) + 14*(c7+c14) + 20*(c11) + 32*(c23) + 44*(c10))*u1.x*u1.y +
                 ((c1+c3+c6+c8) + 2*(c4+c26+c28) + 4*(c2) + 5*(c13+c15) + 6*(c20) + 8*(c7) + 9*(c22+c24) + 12*(c27) + 14*(c9+c11) + 20*(c14) + 32*(c23) + 44*(c10))*u1.x*u1.z +
                 ((c2+c3+c34+c35) + 2*(c9+c22) + 3*(c8+c15+c21+c28) + 5*(c7+c14+c20+c27) + 11*(c11+c24) + 17*(c10+c23))*u2.y +
                 ((c3+c4+c8+c13+c21+c26+c35+c36) + 2*(c2+c34) + 6*(c15+c28) + 7*(c7+c9+c20+c22) + 16*(c11+c14+c24+c27) + 38*(c10+c23))*u1.y*u1.z +
                 ((c2+c4+c34+c36) + 2*(c7+c20) + 3*(c13+c15+c26+c28) + 5*(c9+c11+c22+c24) + 11*(c14+c27) + 17*(c10+c23))*u2.z))*u1.w;
}

__inline float eval_M_expr_green(float* c, float4 u1, float4 u2, float4 u3)
{
    return (12*((c0+c1+c20+c22) + 2*(c8+c13) + 3*(c3+c4+c24+c27) + 4*(c7+c9+c15) + 8*(c2+c23) + 12*(c11+c14) + 24*(c10))*u1.x +
            24*(((c4+c13+c20+c22) + 2*(c9+c28) + 3*(c3+c8) + 4*(c2+c7+c27) + 6*(c15+c24) + 10*(c14+c23) + 16*(c11) + 22*(c10))*u1.y +
                ((c3+c8+c20+c22) + 2*(c7+c28) + 3*(c4+c13) + 4*(c2+c9+c24) + 6*(c15+c27) + 10*(c11+c23) + 16*(c14) + 22*(c10))*u1.z +
                ((c5+c7+c8+c9+c13+c28) + 2*(c3+c4+c24+c27) + 3*(c2+c23) + 4*(c15) + 7*(c11+c14) + 10*(c10))*u1.w))*u1.x*u1.w;
}

__inline float eval_M_expr_blue(float* c, float4 u1, float4 u2, float4 u3)
{
    return  ( 2*((c2+c3+c4+c5+c7+c8+c9+c12+c13+c16+c17+c18+c20+c21+c22+c25+c26+c29+c30+c31+c34+c35+c36+c37) + 21*(c10+c11+c14+c15+c23+c24+c27+c28))*u2.x +
              3*(((c4+c5+c13+c16+c26+c29+c36+c37) + 3*(c2+c3+c9+c12+c22+c25+c34+c35) + 4*(c7+c8+c20+c21) + 34*(c14+c15+c27+c28) + 50*(c10+c11+c23+c24))*u1.y +
                 ((c3+c5+c8+c18+c21+c31+c35+c37) + 3*(c2+c4+c7+c17+c20+c30+c34+c36) + 4*(c9+c13+c22+c26) + 34*(c11+c15+c24+c28) + 50*(c10+c14+c23+c27))*u1.z +
                 ((c20+c21+c22+c25+c26+c29+c30+c31) + 3*(c7+c8+c9+c12+c13+c16+c17+c18) + 4*(c2+c3+c4+c5) + 34*(c23+c24+c27+c28) + 50*(c10+c11+c14+c15))*u1.w)*u1.x +
             12*(((c2+c3+c9+c12+c22+c25+c34+c35) + 2*(c7+c8+c20+c21) + 6*(c14+c15+c27+c28) + 14*(c10+c11+c23+c24))*u2.y +
                 ((c3+c4+c8+c13+c21+c26+c35+c36) + 2*(c2+c34) + 3*(c7+c9+c20+c22) + 14*(c15+c28) + 20*(c11+c14+c24+c27) + 30*(c10+c23))*u1.y*u1.z +
                 ((c4+c5+c13+c16+c20+c21+c22+c25) + 2*(c9+c12) + 3*(c2+c3+c7+c8) + 14*(c27+c28) + 20*(c14+c15+c23+c24) + 30*(c10+c11))*u1.y*u1.w +
                 ((c2+c4+c7+c17+c20+c30+c34+c36) + 2*(c9+c13+c22+c26) + 6*(c11+c15+c24+c28) + 14*(c10+c14+c23+c27))*u2.z +
                 ((c3+c5+c8+c18+c20+c22+c26+c30) + 2*(c7+c17) + 3*(c2+c4+c9+c13) + 14*(c24+c28) + 20*(c11+c15+c23+c27) + 30*(c10+c14))*u1.z*u1.w +
                 ((c7+c8+c9+c12+c13+c16+c17+c18) + 2*(c2+c3+c4+c5) + 6*(c23+c24+c27+c28) + 14*(c10+c11+c14+c15))*u2.w))*u1.x;
}


float3 eval_G_expr_rgb(float* c, float4 u1, float4 u2)
{
    float3 g;
    g.x = ((c20-c3-c7-c8-c14-c15-c2+c21+c27+c28+c34+c35) + 3*(-c10-c11+c23+c24))*u2.y +
          ((c20-c3-c4-c7-c8-c9-c13+c21+c22+c26+c35+c36) + 2*(-c15-c2+c28+c34) + 4*(-c11-c14+c24+c27) + 6*(-c10+c23))*(u1.y*u1.z) +
          ((c22-c4-c9-c11-c13-c15-c2+c24+c26+c28+c34+c36) + 3*(-c10-c14+c23+c27))*u2.z;
    g.y = 2*((c14-c8-c7+c15-c20-c21+c27+c28))*u2.y +
            ((c4-c3-c8-c9+c13-c21-c22+c26-c35+c36) + 2*(-c11-c24) + 3*(-c7-c20) + 4*(-c10+c15-c23+c28) + 6*(c14+c27))*(u1.y*u1.z) +
            ((c13-c9-c11-c7+c15+c17-c20-c22-c24+c26+c28+c30) + 3*(-c10+c14-c23+c27))*u2.z;    
    g.z =   ((c8-c7-c9+c12-c14+c15-c20+c21-c22+c25-c27+c28) + 3*(-c10+c11-c23+c24))*u2.y +
            ((c3-c4-c7+c8-c13-c20+c21-c26+c35-c36) + 2*(-c14-c27) + 3*(-c9-c22) + 4*(-c10+c15-c23+c28) + 6*(c11+c24))*(u1.y*u1.z) +
          2*((c11-c9-c13+c15-c22+c24-c26+c28))*u2.z;

    return 4*g;
}

float3 eval_G_expr_rg(float* c, float4 u1, float4 u2)
{
    float3 g;
    g.x = ((c20-c1-c3-c4-c0+c22+c24+c27) + 4*(-c2+c23))*u1.x +
          ((-c4-c6-c8-c13-c15-c0) + 2*(-c3-c7-c14+c21+c22+c28) + 3*(c20+c27) + 4*(-c2-c10-c11+c24) + 8*(c23))*u1.y +
          ((-c3-c6-c8-c13-c15-c1) + 2*(c20-c4-c9-c11+c26+c28) + 3*(c22+c24) + 4*(-c2-c10-c14+c27) + 8*(c23))*u1.z;
    g.y = ((c4-c0-c6-c8+c13+c15-c20+c27) + 4*(-c7+c14))*u1.x +
          ((c4-c0-c6+c13) + 2*(-c21+c28) + 3*(-c8+c15-c20+c27) + 6*(-c7+c14))*u1.y +
          ((-c3-c1-c6-c8-c22-c24) + 2*(c4-c9-c11-c20+c26+c28) + 3*(c13+c15) + 4*(-c7-c10-c23+c27) + 8*(c14))*u1.z;
    g.z = ((c3-c1-c6+c8-c13+c15-c22+c24) + 4*(-c9+c11))*u1.x +
          ((-c0-c4-c6-c13-c20-c27) + 2*(c3-c7-c14+c21-c22+c28) + 3*(c8+c15) + 4*(-c9-c10-c23+c24) + 8*(c11))*u1.y +
          ((c3-c1-c6+c8) + 2*(-c26+c28) + 3*(-c13+c15-c22+c24) + 6*(-c9+c11))*u1.z;
    
    return (4*u1.x)*g;
}

float3 eval_G_expr_gb(float* c, float4 u1, float4 u2)
{
    float3 g;
    g.x =   ((c20-c4-c5-c7-c8-c13-c16+c21+c22+c25) + 2*(-c14-c15) + 3*(-c3-c2) + 4*(-c10-c11+c27+c28) + 6*(c23+c24))*u1.y +
            ((c20-c3-c5-c8-c9-c13-c18+c22+c26+c30) + 2*(-c11-c15) + 3*(-c4-c2) + 4*(-c10-c14+c24+c28) + 6*(c23+c27))*u1.z +
          2*((c23-c3-c4-c5-c2+c24+c27+c28))*u1.w ;
    g.y = ((c4-c3-c2+c5+c13+c16-c20-c21-c22-c25) + 2*(-c23-c24) + 3*(-c7-c8) + 4*(-c10-c11+c27+c28) + 6*(c14+c15))*u1.y +
          ((c4-c3-c2+c5-c8-c9+c13+c18-c20-c22+c26+c30) + 2*(-c7+c17-c24+c28) + 4*(-c11+c15-c23+c27) + 6*(-c10+c14))*u1.z +
          ((c4-c3-c2+c5-c7-c8+c17+c18-c23-c24+c27+c28) + 3*(-c10-c11+c14+c15))*u1.w ;
    g.z = ((c3-c2-c4+c5-c7+c8-c13+c16-c20+c21-c22+c25) + 2*(-c9+c12-c27+c28) + 4*(-c14+c15-c23+c24) + 6*(-c10+c11))*u1.y +
          ((c3-c2-c4+c5+c8+c18-c20-c22-c26-c30) + 2*(-c23-c27) + 3*(-c9-c13) + 4*(-c10-c14+c24+c28) + 6*(c11+c15))*u1.z +
          ((c3-c2-c4+c5-c9+c12-c13+c16-c23+c24-c27+c28) + 3*(-c10+c11-c14+c15))*u1.w;

    return (4*u1.w)*g;
}

float3 eval_G_expr_red(float* c, float4 u1, float4 u2)
{
    float3 g;
    
    g.x = 4*((c19-c1-c3-c4-c6-c8-c13-c15-c0+c21+c26+c28) + 2*(-c7-c9-c11-c14) + 3*(c20+c22+c24+c27) + 4*(-c2-c10) + 8*(c23))*u1.x +
          2*(((c19-c4-c6-c13-c0+c26+c32+c36) + 2*(-c3-c9+c22+c35) + 3*(-c8-c15+c21+c28) + 4*(-c2+c34) + 5*(-c7-c14+c20+c27) + 8*(-c11+c24) + 12*(-c10+c23))*u1.y +
             ((c19-c3-c6-c8-c1+c21+c33+c35) + 2*(-c4-c7+c20+c36) + 3*(-c13-c15+c26+c28) + 4*(-c2+c34) + 5*(-c9-c11+c22+c24) + 8*(-c14+c27) + 12*(-c10+c23))*u1.z) +
            ((-c1-c3-c4-c0+c32+c33+c35+c36) + 2*(c19-c6-c8-c13-c15+c21+c26+c28) + 4*(-c2+c34) + 5*(-c7-c9-c11-c14+c20+c22+c24+c27) + 12*(-c10+c23))*u1.w;
    g.y = 4*((c4-c0-c19-c21+c26+c28) + 2*(-c6-c8+c13+c15) + 3*(-c20+c27) + 6*(-c7+c14))*u1.x +
          2*(((c4-c0-c6+c13-c19+c26-c32+c36) + 5*(-c8+c15-c21+c28) + 9*(-c7+c14-c20+c27))*u1.y +
             ((-c3-c1-c6-c8-c19-c21-c33-c35) + 2*(c4+c36) + 3*(-c9-c11-c22-c24) + 5*(c13+c15+c26+c28) + 6*(-c7-c20) + 8*(-c10-c23) + 12*(c14+c27))*u1.z) +
            ((c4-c0-c32+c36) + 3*(-c6-c8+c13+c15-c19-c21+c26+c28) + 9*(-c7+c14-c20+c27))*u1.w;
    g.z = 4*((c3-c1-c19+c21-c26+c28) + 2*(-c6+c8-c13+c15) + 3*(-c22+c24) + 6*(-c9+c11))*u1.x +
          2*(((-c0-c4-c6-c13-c19-c26-c32-c36) + 2*(c3+c35) + 3*(-c7-c14-c20-c27) + 5*(c8+c15+c21+c28) + 6*(-c9-c22) + 8*(-c10-c23) + 12*(c11+c24))*u1.y +
             ((c3-c1-c6+c8-c19+c21-c33+c35) + 5*(-c13+c15-c26+c28) + 9*(-c9+c11-c22+c24))*u1.z) +
            ((c3-c1-c33+c35) + 3*(-c6+c8-c13+c15-c19+c21-c26+c28) + 9*(-c9+c11-c22+c24))*u1.w;

    return u1.w*g;
}


float3 eval_G_expr_green(float* c, float4 u1, float4 u2)
{
    float3 g;
    
    g.x = (c20-c1-c0+c22) + 2*(-c5+c28) + 3*(-c3-c4+c24+c27) + 6*(-c2+c23);
    g.y = (-c1-c3-c0-c20-c22-c24) + 2*(-c2+c5-c8+c13-c23+c28) + 3*(c4+c27) + 4*(-c7-c10-c11+c15) + 8*(c14);
    g.z = (-c1-c0-c4-c20-c22-c27) + 2*(-c2+c5+c8-c13-c23+c28) + 3*(c3+c24) + 4*(-c9-c10-c14+c15) + 8*(c11);

    return (4*u1.x*u1.w)*g;
}


float3 eval_G_expr_blue(float* c, float4 u1, float4 u2)
{
    float3 g;
    g.x =   ((c20-c7-c8-c9-c12-c13-c16-c17-c18+c21+c22+c25+c26+c29+c30+c31) + 2*(-c3-c4-c5-c2+c34+c35+c36+c37) + 8*(-c10-c11-c14-c15+c23+c24+c27+c28))*u1.x +
          2*(((-c4-c5-c9-c12-c13-c16+c22+c25+c26+c29+c36+c37) + 2*(c20-c7-c8+c21) + 3*(-c3-c2+c34+c35) + 6*(-c14-c15+c27+c28) + 10*(-c10-c11+c23+c24))*u1.y +
             ((c20-c3-c5-c7-c8-c17-c18+c21+c30+c31+c35+c37) + 2*(-c9-c13+c22+c26) + 3*(-c4-c2+c34+c36) + 6*(-c11-c15+c24+c28) + 10*(-c10-c14+c23+c27))*u1.z +
             ((c20-c7-c8-c9-c12-c13-c16-c17-c18+c21+c22+c25+c26+c29+c30+c31) + 4*(-c3-c4-c5-c2) + 6*(-c10-c11-c14-c15) + 10*(c23+c24+c27+c28))*u1.w);
    g.y =  ((c4-c3-c2+c5-c9-c12+c13+c16-c22-c25+c26+c29-c34-c35+c36+c37) + 2*(-c7-c8+c17+c18-c20-c21+c30+c31) + 8*(-c10-c11+c14+c15-c23-c24+c27+c28))*u1.x +
         2*(((c4-c3-c2+c5-c9-c12+c13+c16-c22-c25+c26+c29-c34-c35+c36+c37) + 4*(-c7-c8-c20-c21) + 6*(-c10-c11-c23-c24) + 10*(c14+c15+c27+c28))*u1.y +
            ((c4-c3-c2+c5-c8+c18-c21+c31-c34-c35+c36+c37) + 2*(-c9+c13-c22+c26) + 3*(-c7+c17-c20+c30) + 6*(-c11+c15-c24+c28) + 10*(-c10+c14-c23+c27))*u1.z +
            ((-c9-c12+c13+c16-c20-c21-c22-c25+c26+c29+c30+c31) + 2*(c4-c3-c2+c5) + 3*(-c7-c8+c17+c18) + 6*(-c23-c24+c27+c28) + 10*(-c10-c11+c14+c15))*u1.w);
    g.z =   ((c3-c2-c4+c5-c7+c8-c17+c18-c20+c21-c30+c31-c34+c35-c36+c37) + 2*(-c9+c12-c13+c16-c22+c25-c26+c29) + 8*(-c10+c11-c14+c15-c23+c24-c27+c28))*u1.x +
          2*(((c3-c2-c4+c5-c13+c16-c26+c29-c34+c35-c36+c37) + 2*(-c7+c8-c20+c21) + 3*(-c9+c12-c22+c25) + 6*(-c14+c15-c27+c28) + 10*(-c10+c11-c23+c24))*u1.y +
             ((c3-c2-c4+c5-c7+c8-c17+c18-c20+c21-c30+c31-c34+c35-c36+c37) + 4*(-c9-c13-c22-c26) + 6*(-c10-c14-c23-c27) + 10*(c11+c15+c24+c28))*u1.z +
             ((-c7+c8-c17+c18-c20+c21-c22+c25-c26+c29-c30+c31) + 2*(c3-c2-c4+c5) + 3*(-c9+c12-c13+c16) + 6*(-c23+c24-c27+c28) + 10*(-c10+c11-c14+c15))*u1.w);
        
    return u1.x*g;
}

__inline void fetch_coefficients(float* c, image3d_t vol, int3 org, int3 R, int3 vecPx, int3 vecPy, int3 vecPz)
{
    int3 dirx = R*vecPx;
    int3 diry = R*vecPy;
    int3 dirz = R*vecPz;

    int3 offset = org;

    c10 = read_imagef(vol, sp2, (int4)(offset,1)).r; //( 0, 0, 0)
    offset += dirx;
    c23 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -dirz;
    c22 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirx;
    c33 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirz;
    c34 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -diry;
    c32 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -dirx;
    c20 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -dirz;
    c19 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -dirx;
    c6 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirz;
    c7 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -dirx;
    c0 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += diry;
    c2 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -dirz;
    c1 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirx;
    c9 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += diry;
    c13 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirx;
    c26 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirz;
    c27 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirx;
    c36 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -3*dirx;
    c4 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirx;
    c14 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += diry;
    c17 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirx;
    c30 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirz;
    c31 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -dirx;
    c18 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -diry;
    c15 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -dirx;
    c5 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -diry;
    c3 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirx;
    c11 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -diry;
    c8 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirx;
    c21 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += diry;
    c24 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirx;
    c35 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += diry;
    c37 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -dirx;
    c28 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirz;
    c29 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -dirx;
    c16 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += -diry;
    c12 = read_imagef(vol, sp2, (int4)(offset,1)).r;
    offset += dirx;
    c25 = read_imagef(vol, sp2, (int4)(offset,1)).r;  
}


__inline float4 to_barycentric(int type_tet, float3 p)
{
    return     convert_float(type_tet==TYPE_BLUE)*2.0f*
                (float4)( p.x+p.y+p.z-1.0f,
                             -p.y    +0.5f,
                                 -p.z+0.5f,
                         -p.x        +0.5f) +
               convert_float(type_tet==TYPE_GREEN)*
                (float4)(-p.x-p.y-p.z+1.0f,
                          p.x-p.y+p.z     ,
                          p.x+p.y-p.z     ,
                         -p.x+p.y+p.z     ) +
               convert_float(type_tet==TYPE_RED)*2.0f*
                (float4)(-p.x        +0.5f,
                                  p.z     ,
                              p.y         ,
                          p.x-p.y-p.z     );    

}

__inline float eval(float3 p_in,  __read_only image3d_t vol)
{
    int3 org = convert_int3(round(p_in));
    float3 p_local = p_in - convert_float3(org);

    int3 R = 2*(int3)(p_local.x>0, p_local.y>0, p_local.z>0)-1;

    float3 p_cube = p_local.xyz*convert_float3(R);
    int4   bit = (int4)( p_cube.x-p_cube.y-p_cube.z>0,
                        -p_cube.x+p_cube.y-p_cube.z>0,
                        -p_cube.x-p_cube.y+p_cube.z>0,
                         p_cube.x+p_cube.y+p_cube.z>1);
    // bit_tet   type_tet type_P permutation
    // 0 1 2 3
    // -------------------------------------
    // 1 0 0 0       2      0        123    (edge/red)  xyz
    // 0 1 0 0       2      1        231    (edge/red)  yzx
    // 0 0 1 0       2      2        312    (edge/red)  zxy
    // 0 0 0 1       0      0        123    (oct/blue)
    // 0 0 0 0       1      0        123    (vert/green)
    int type_tet = bit.x+bit.y+bit.z-bit.w+1;
    int type_P = 2*bit.z + bit.y; // one of three even permutations

    int3 vecPx = (int3)(type_P==0, type_P==1, type_P==2);
    int3 vecPy = vecPx.zxy;
    int3 vecPz = vecPx.yzx;

    float3 p_ref = p_cube;
    if(type_P==1) p_ref = p_ref.yzx;
    else if(type_P==2) p_ref= p_ref.zxy;

    float c[38];
    fetch_coefficients(c, vol, org, R, vecPx, vecPy, vecPz);

    float4 u1 = to_barycentric(type_tet, p_ref);
    float4 u2 = u1*u1;
    float4 u3 = u2*u1;
    float val = eval_M_expr_rgb(c, u1, u2, u3);

    if(type_tet==TYPE_RED)   val += eval_M_expr_red(c, u1, u2, u3);
    else                     val += eval_M_expr_gb(c, u1, u2, u3);
    if(type_tet==TYPE_GREEN) val += eval_M_expr_green(c, u1, u2, u3);
    if(type_tet==TYPE_BLUE)  val += eval_M_expr_blue(c, u1, u2, u3);
    else                     val += eval_M_expr_rg(c, u1, u2, u3);
    return DENOM_M*val;
}

__inline float3 eval_g(float3 p_in,  __read_only image3d_t vol)
{
    int3 org = convert_int3(round(p_in));
    float3 p_local = p_in - convert_float3(org);

    int3 R = 2*(int3)(p_local.x>0, p_local.y>0, p_local.z>0)-1;

    float3 p_cube = p_local.xyz*convert_float3(R);
    int4   bit = (int4)( p_cube.x-p_cube.y-p_cube.z>0,
                        -p_cube.x+p_cube.y-p_cube.z>0,
                        -p_cube.x-p_cube.y+p_cube.z>0,
                         p_cube.x+p_cube.y+p_cube.z>1);
    // bit_tet   type_tet type_P permutation
    // 0 1 2 3
    // -------------------------------------
    // 1 0 0 0       2      0        123    (edge/red)  xyz
    // 0 1 0 0       2      1        231    (edge/red)  yzx
    // 0 0 1 0       2      2        312    (edge/red)  zxy
    // 0 0 0 1       0      0        123    (oct/blue)
    // 0 0 0 0       1      0        123    (vert/green)
    int type_tet = bit.x+bit.y+bit.z-bit.w+1;
    int type_P = 2*bit.z + bit.y; // one of three even permutations


    int3 vecPx = (int3)(type_P==0, type_P==1, type_P==2);
    int3 vecPy = vecPx.zxy;
    int3 vecPz = vecPx.yzx;

    float3 p_ref = p_cube;
    if(type_P==1) p_ref = p_ref.yzx;
    else if(type_P==2) p_ref= p_ref.zxy;    

    float c[38];
    fetch_coefficients(c, vol, org, R, vecPx, vecPy, vecPz);

    float4 u1 = to_barycentric(type_tet, p_ref);
    float4 u2 = u1*u1;

    float3 g = eval_G_expr_rgb(c, u1, u2);
   
    if(type_tet==TYPE_RED)     g += eval_G_expr_red(c, u1, u2);
    else                       g += eval_G_expr_gb(c, u1, u2);
    if(type_tet==TYPE_GREEN)   g += eval_G_expr_green(c, u1, u2);
    if(type_tet==TYPE_BLUE)    g += eval_G_expr_blue(c, u1, u2);
    else                       g += eval_G_expr_rg(c, u1, u2);

    if(type_P == 1) g = g.zxy;
    else if(type_P == 2) g = g.yzx;
    g *= convert_float3(R);
    
    return DENOM_G*g;
}
