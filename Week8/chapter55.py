def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size)//stride +1

H,W= 4,4
kh, kw = 3,3
sh, sw = 1,1
ph,pw = 1,1

oh = get_conv_outsize(H,kh,sh,ph)
ow = get_conv_outsize(W,kw,sw,pw)
print(oh,ow)