import math

def oversample(all_paths, per_class_split, oversample_ids, class_names):
  union = set()
  all_sum = 0
  print('Oversample stats:')
  print('Total images before =', len(all_paths[0]))
  for i in oversample_ids:
    duplicates = 1
    print(f'id = {i} -> {class_names[i]} : num of oversampled =', len(per_class_split[i]))
    all_sum += len(per_class_split[i])
    for idx in per_class_split[i]:
      if idx not in union:
        union.add(idx)
      for j in range(duplicates):
        for paths in all_paths:
          paths.append(paths[idx])
  print('Total oversampled =', all_sum, '/ union =', len(union))
  print('Total images after =', len(all_paths[0]))


def oversample_end(all_paths, num):
  for paths in all_paths:
    oversample = []
    for i in range(num):
      oversample.append(paths[-1-i])
    paths.extend(oversample)


def print_class_colors(dataset):
  for color, name in zip(dataset.class_colors, dataset.class_names):
    print(color, '\t', name)


def get_pyramid_loss_scales(downsampling_factor, upsampling_factor):
  num_scales = int(math.log2(downsampling_factor // upsampling_factor))
  scales = [downsampling_factor]
  for i in range(num_scales - 1):
    assert scales[-1] % 2 == 0
    scales.append(scales[-1] // 2)
  return scales


def get_data_bound(dataset):
  min_val = (-dataset.mean.max()) / dataset.std.min()
  max_val = (255-dataset.mean.min()) / dataset.std.min()
  return float(min_val), float(max_val)
