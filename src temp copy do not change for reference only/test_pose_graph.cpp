#include "visualOdometry.h"
#include <iostream>
#include <cmath>

int main()
{
    std::vector<PoseGraphNode3D> nodes(2);
    nodes[0].x = 0.0;
    nodes[0].y = 0.0;
    nodes[0].z = 0.0;
    nodes[0].yaw = 0.0;

    nodes[1].x = 10.0;
    nodes[1].y = 0.0;
    nodes[1].z = 0.0;
    nodes[1].yaw = 0.0;

    std::vector<PoseGraphEdge3D> edges;
    edges.push_back(PoseGraphEdge3D{0, 1, 1.0, 0.0, 0.0, 0.0});

    optimizePoseGraph(nodes, edges, 10);

    const double error = std::abs(nodes[1].x - 1.0) + std::abs(nodes[1].y) + std::abs(nodes[1].z) + std::abs(nodes[1].yaw);
    if (error > 0.02)
    {
        std::cerr << "Pose graph optimization failed: error=" << error << std::endl;
        return 1;
    }

    std::cout << "Pose graph optimization ok (error=" << error << ")" << std::endl;
    return 0;
}
