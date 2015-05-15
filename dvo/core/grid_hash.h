#pragma once
#include <vector>

namespace dvo {

template<typename HashElement = unsigned int>
struct GridHashBase {
  GridHashBase() = delete;
  /// @size (width, height)
  GridHashBase(unsigned int width, unsigned int height, unsigned int grid_cell_size) 
    : grid_cell_size(grid_cell_size) {
    assert(grid_cell_size < width && grid_cell_size < height);
    grid_width = width / grid_cell_size;
    if (width % grid_cell_size == 1)
      grid_width += 1;

    grid_height = height / grid_cell_size;
    if (height % grid_cell_size == 1)
      grid_height += 1;

    hash_table.resize((grid_width+1)*(grid_height+1));
  }

  unsigned int hash_index(unsigned int x, unsigned int y) {
    x /= grid_cell_size;
    y /= grid_cell_size;
    return y*grid_width + x;
  }

  void insert(unsigned int x, unsigned int y, HashElement el) {
    auto idx = hash_index(x, y);
    hash_table[idx].push_back(el);
  }

  std::vector<HashElement> max_elements() const {
    std::vector<HashElement> els;
    els.reserve(hash_table.size());
    for (auto& t : hash_table) {
      if (t.empty()) continue;
      auto e = *std::max_element(t.begin(), t.end());
      els.push_back(e);
    }
    return els;
  }

  unsigned int grid_cell_size;
  unsigned int grid_width, grid_height;
  std::vector<std::vector<HashElement>> hash_table;
};

using GridHash = GridHashBase<unsigned int>;

} // namespace dvo