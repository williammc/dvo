#pragma once
namespace dvo {
namespace util {

template<typename T>
class Revertable {
 public:
  Revertable() : value() {
  }
  Revertable(const T& value) : value(value) {
  }

  inline const T& operator()() const {
    return value;
  }

  T& update() {
    old = value;

    return value;
  }

  void revert() {
    value = old;
  }
private:
  T old, value;
};

}  // namespace util
}  // namespace dvo