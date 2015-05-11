// Copyright 2014 The DVO Authors. All rights reserved.
#include "dvo/util/id_register.h"
#include <sstream>

namespace dvo {
static IdType incr_index = 0;

IdRegister::IdRegister() {}

IdRegister::~IdRegister() {}

void IdRegister::Reset(IdType id) {
  incr_index = id;
}

IdType IdRegister::Register() {
  incr_index += 1;
  return incr_index;
}

IdType IdRegister::Next() {
  return incr_index + 1;
}

}  // namespace dvo