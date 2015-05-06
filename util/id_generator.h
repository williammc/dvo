#pragma once
#include <string>
#include <vector>
#include <sstream>

namespace dvo {
namespace util {

class IdGenerator {
public:
  IdGenerator(const std::string prefix) :
    prefix_(prefix),
    var_(0) {
  }

  const std::vector<std::string>& all() {
    return generated_;
  }

  void next(std::string& id) {
    id = next();
  }

  std::string next() {
    std::stringstream ss;
    ss << prefix_ << var_;

    var_ += 1;
    generated_.push_back(ss.str());

    return ss.str();
  }

  void reset() {
    var_ = 0;
    generated_.clear();
  }
private:
  std::string prefix_;
  std::vector<std::string> generated_;
  int var_;
};

}  // namespace util
}  // namespace dvo