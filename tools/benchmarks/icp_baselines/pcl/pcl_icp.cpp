// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PCL ICP baseline for the rigid-registration comparison. Reads two XYZ point
// clouds, runs point-to-point ICP and Generalized ICP (each best-of-three), and
// prints "RESULT <name> <16 row-major transform floats> <ms>" per method.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

Cloud::Ptr readxyz(const char* path) {
  Cloud::Ptr c(new Cloud); std::ifstream f(path); double x,y,z;
  while (f >> x >> y >> z) c->push_back(pcl::PointXYZ(x,y,z));
  return c;
}
void print_result(const char* name, const Eigen::Matrix4f& T, double ms) {
  printf("RESULT %s ", name);
  for (int i=0;i<4;i++) for (int j=0;j<4;j++) printf("%.7f ", T(i,j));
  printf("%.3f\n", ms);
}
int main(int argc, char** argv) {
  double mcd = std::atof(argv[3]);
  auto src = readxyz(argv[1]); auto tgt = readxyz(argv[2]);
  // point-to-point ICP
  {
    double best=1e18; Eigen::Matrix4f T;
    for (int r=0;r<3;r++){
      pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
      icp.setInputSource(src); icp.setInputTarget(tgt);
      icp.setMaxCorrespondenceDistance(mcd); icp.setMaximumIterations(50);
      icp.setTransformationEpsilon(1e-10);
      Cloud out; auto t0=std::chrono::high_resolution_clock::now();
      icp.align(out); auto t1=std::chrono::high_resolution_clock::now();
      double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
      if (ms<best){best=ms; T=icp.getFinalTransformation();}
    }
    print_result("pcl_point_to_point", T, best);
  }
  // Generalized ICP (plane-to-plane)
  {
    double best=1e18; Eigen::Matrix4f T;
    for (int r=0;r<3;r++){
      pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> gicp;
      gicp.setInputSource(src); gicp.setInputTarget(tgt);
      gicp.setMaxCorrespondenceDistance(mcd); gicp.setMaximumIterations(50);
      Cloud out; auto t0=std::chrono::high_resolution_clock::now();
      gicp.align(out); auto t1=std::chrono::high_resolution_clock::now();
      double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
      if (ms<best){best=ms; T=gicp.getFinalTransformation();}
    }
    print_result("pcl_gicp", T, best);
  }
  return 0;
}
