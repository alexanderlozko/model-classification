from model.Model_new import Model

a = Model('../data/category_with_received_data.csv', '../data/category_with_received_data.csv', 64, 8)

print(a.vocab)
a.vocab_size = 1
print(a.vocab_size)