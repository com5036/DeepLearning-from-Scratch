from layer_naive import *

apple_num = 2
orange_num = 3
apple = 100
orange = 150
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple_num, apple)
orange_price = mul_orange_layer.forward(orange_num, orange)
total_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(total_price, tax)

# backward
dprice = 1
dtotal_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dtotal_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_apple_layer.backward(dorange_price)

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dApple:", dorange)
print("dApple_num:", int(dorange_num))
print("dTax:", dtax)

