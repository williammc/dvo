#pragma once

/// Macro to easily define fluent interfaces.
#define FI_ATTRIBUTE(FI_TYPE, ATTR_TYPE, ATTR_NAME) \
  protected: \
    ATTR_TYPE ATTR_NAME ## _; \
  public: \
    FI_TYPE& ATTR_NAME(ATTR_TYPE const& value) \
    { \
      ATTR_NAME ## _ = value; \
      return *this; \
    } \
    ATTR_TYPE const& ATTR_NAME() const \
    { \
      return ATTR_NAME ## _; \
    } \
    ATTR_TYPE& ATTR_NAME() \
    { \
      return ATTR_NAME ## _; \
    }

