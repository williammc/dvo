#pragma once
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#ifdef HAVE_BOOST_LOG
#include <boost/log/trivial.hpp>
#else
#include <stdio.h>
#define BOOST_LOG_TRIVIAL(param) std::cout
#endif
namespace fs = boost::filesystem;

namespace dvo {
namespace util {
enum LSMode {LSMODE_FILES = 0, LSMODE_FOLDERS=1, LSMODE_ALL=2};
// similar to ls command
inline std::vector<fs::path> ls(const std::string &path,
                                LSMode mode = LSMODE_FILES) {
  fs::path full_path( fs::initial_path<fs::path>() );
  full_path = fs::system_complete( fs::path( path ) );

  std::vector<fs::path> files(0);
  std::vector<fs::path> dirs(0);
  std::vector<fs::path> others(0);

  try {
    if (!fs::exists(full_path))
      BOOST_LOG_TRIVIAL(fatal) << "ls: path doesn't exists";

    if (fs::is_directory(full_path)) {
      fs::directory_iterator end_iter;
      for (fs::directory_iterator dir_itr( full_path );
           dir_itr != end_iter; ++dir_itr) {
        try {
          if (fs::is_directory(dir_itr->status()))
            dirs.push_back(dir_itr->path()); // .file_string()
          else if (fs::is_regular_file( dir_itr->status()))
            files.push_back(dir_itr->path());
          else
            others.push_back(dir_itr->path());
        } catch (const fs::filesystem_error & e) {
          BOOST_LOG_TRIVIAL(error) << "Filesystem error: "
                     << e.path1().string() << " " << e.what();
        }
      }
    } else {
      if (fs::is_regular_file( full_path)) {
        BOOST_LOG_TRIVIAL(debug) << "ls: given path is a file, returning file name";
        files.push_back(full_path.string());
        return files;
      }
      BOOST_LOG_TRIVIAL(fatal) << "ls: path is not a directory nor regular file";
    }
  } catch(fs::filesystem_error &e) {
    BOOST_LOG_TRIVIAL(error) << e.what();
    BOOST_LOG_TRIVIAL(fatal) << "ls: Can't read given path";
  }

  switch (mode) {
  case LSMODE_FILES:
    return files;

  case LSMODE_FOLDERS:
    return dirs;

  case LSMODE_ALL:
    // append files and others to to directory list
    dirs.insert(dirs.end(),files.begin(),files.end());
    dirs.insert(dirs.end(),others.begin(),others.end());
    return dirs;
  }
}

inline bool FilterExtension(fs::path const& p, std::string const &ext) {
  unsigned const   len = ext.length();
  std::string s = p.extension().string();
  if (s.length() < len) return false;
  return s.compare(s.length() - len, len, ext) == 0;
}

inline void ScanDirectory(const std::string &path,
                          const std::string &extension,
                          std::vector<fs::path>& entries) {

  std::vector<fs::path> files = ls(path);
  for (size_t i = 0; i < files.size(); ++i)
    if (FilterExtension(files[i],extension))
      entries.push_back(files[i]);

  std::sort(entries.begin(), entries.end());
}

inline bool FileExists(const std::string& filename) {
  try {
    fs::path full_path = fs::system_complete(fs::path(filename));
    return (fs::exists(full_path));
  } catch (fs::filesystem_error &e) {
    BOOST_LOG_TRIVIAL(error) << "filesystem error: filename: " << filename
               << " fs::error: " << e.what();
  }
  return false;
}

inline bool CreateDirectory(const std::string &path_name) {
  fs::path p = fs::system_complete(fs::path( path_name));
  try {
    if (boost::filesystem::create_directory(p))
      return true;
  } catch (boost::filesystem::filesystem_error &e) {
    BOOST_LOG_TRIVIAL(error) << "filesystem error: Couldn't create directory: "
               << path_name << " " << e.what();
  }
  return false;
}

inline bool RemoveDirectory(const std::string &path_name) {
  fs::path p = fs::system_complete(fs::path(path_name));
  try {
    if (boost::filesystem::remove(p))
      return true;
  } catch (boost::filesystem::filesystem_error &e) {
    BOOST_LOG_TRIVIAL(error) << "filesystem error: Couldn't remove directory: "
               << path_name << " " << e.what();
  }
  return false;
}

inline bool RemoveFile(const std::string &path_name) {
  fs::path p = fs::system_complete(fs::path(path_name));
  try {
    if (boost::filesystem::remove(p))
      return true;
  }catch (boost::filesystem::filesystem_error &e) {
    BOOST_LOG_TRIVIAL(error) << "filesystem error: Couldn't remove file: "
               << path_name << " " << e.what();
  }
  return false;
}
}  // namespace util
}  // namespace dvo