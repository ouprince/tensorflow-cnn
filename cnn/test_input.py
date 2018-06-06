from input_data import data_load

xs,ys = data_load("train.raw.data.test")

print len(ys),len(xs)

for i in ys:
    print i
