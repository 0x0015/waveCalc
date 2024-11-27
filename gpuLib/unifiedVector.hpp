#pragma once
#include "unifiedAllocator.hpp"
#include <vector>

template<class T> using unified_vector = std::vector<T, unifiedAllocator<T>>;

