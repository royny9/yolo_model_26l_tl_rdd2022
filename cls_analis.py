import os
import matplotlib.pyplot as plt

dir_path = 'RDD_SPLIT/val/labels'
dir_labels = os.listdir(dir_path)

hash_cls_card = dict()

for filename in dir_labels:
    if not filename.endswith('.txt'): continue
    
    with open(os.path.join(dir_path, filename), 'r') as file:
        for line in file:
            line = line.strip() # Убираем лишние пробелы и переносы \n
            if not line:
                continue 
            
            parts = line.split()
            cls_idx = parts[0]
            
            hash_cls_card[cls_idx] = hash_cls_card.get(cls_idx, 0) + 1

print("Статистика классов:", hash_cls_card)


sorted_keys = sorted(hash_cls_card.keys(), key=int)
sorted_values = [hash_cls_card[k] for k in sorted_keys]

plt.bar(sorted_keys, sorted_values, color='skyblue')
plt.xlabel('Класс (ID)')
plt.ylabel('Количество объектов')
plt.title('Распределение дефектов в датасете')
plt.show()

'''очень сильный дисбаланс класса 0
4 класс - износ разметки не учитываем D44 '''