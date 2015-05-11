// Copyright 2014 The DVO Authors. All rights reserved.
#pragma once
#include <cstdint>
#include "dvo/dvo_api.h"

namespace dvo {

using IdType = int;

struct DVO_API IdRegister {
  IdRegister();
  ~IdRegister();

  static void Reset(IdType id = 0);

  /// Register an index value
  static IdType Register();

  /// next index value
  static IdType Next();

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    IdType id;
    id = Next() - 1;
    ar& id;
    Reset(id);
  }
};

}  // namespace dvo
