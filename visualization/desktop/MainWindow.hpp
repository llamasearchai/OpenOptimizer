#pragma once

#include <QMainWindow>
#include <QGraphicsScene>
#include <memory>

#include "ir/graph/ComputationGraph.hpp"

namespace Ui {
class MainWindow;
}

namespace openoptimizer {
namespace visualization {

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    
    void setComputationGraph(std::shared_ptr<ir::ComputationGraph> graph);
    void updateVisualization();

private slots:
    void onImportModel();
    void onRunOptimization();
    void onGenerateCode();
    void onNodeSelected(const QString& nodeName);

private:
    void setupUi();
    void setupConnections();
    void visualizeGraph();
    
    std::unique_ptr<Ui::MainWindow> ui;
    std::shared_ptr<ir::ComputationGraph> graph_;
    QGraphicsScene graphScene_;
};

} // namespace visualization
} // namespace openoptimizer