#include "visualization/desktop/MainWindow.hpp"
#include "ui_MainWindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QPushButton>
#include <QGraphicsItem>

#include <spdlog/spdlog.h>

namespace openoptimizer {
namespace visualization {

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow) {
    setupUi();
    setupConnections();
    spdlog::info("Visualization UI initialized");
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUi() {
    ui->setupUi(this);
    ui->graphView->setScene(&graphScene_);
}

void MainWindow::setupConnections() {
    connect(ui->actionImport, &QAction::triggered, this, &MainWindow::onImportModel);
    connect(ui->actionOptimize, &QAction::triggered, this, &MainWindow::onRunOptimization);
    connect(ui->actionGenerateCode, &QAction::triggered, this, &MainWindow::onGenerateCode);
}

void MainWindow::setComputationGraph(std::shared_ptr<ir::ComputationGraph> graph) {
    graph_ = graph;
    updateVisualization();
}

void MainWindow::updateVisualization() {
    if (graph_) {
        visualizeGraph();
    }
}

void MainWindow::visualizeGraph() {
    graphScene_.clear();
    
    // Create a visualization of the computation graph
    // This is a simplified visualization - a real implementation would use
    // proper graph layout algorithms and interactive elements
    
    const int nodeWidth = 120;
    const int nodeHeight = 60;
    const int xSpacing = 150;
    const int ySpacing = 100;
    
    // Create a mapping of nodes to positions
    std::unordered_map<std::string, QPointF> nodePositions;
    
    // For simplicity, we'll do a layered layout
    std::vector<std::vector<std::shared_ptr<ir::Node>>> layers;
    
    // First layer is input nodes
    layers.push_back(graph_->getInputNodes());
    
    // BFS to build layers
    std::unordered_set<std::shared_ptr<ir::Node>> visited;
    for (const auto& node : graph_->getInputNodes()) {
        visited.insert(node);
    }
    
    while (true) {
        std::vector<std::shared_ptr<ir::Node>> nextLayer;
        
        for (const auto& layer : layers) {
            for (const auto& node : layer) {
                for (const auto& output : node->getOutputs()) {
                    if (visited.find(output) == visited.end()) {
                        nextLayer.push_back(output);
                        visited.insert(output);
                    }
                }
            }
        }
        
        if (nextLayer.empty()) {
            break;
        }
        
        layers.push_back(nextLayer);
    }
    
    // Position nodes based on layers
    for (size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx) {
        const auto& layer = layers[layerIdx];
        
        for (size_t nodeIdx = 0; nodeIdx < layer.size(); ++nodeIdx) {
            const auto& node = layer[nodeIdx];
            
            float x = layerIdx * xSpacing;
            float y = nodeIdx * ySpacing;
            
            nodePositions[node->getName()] = QPointF(x, y);
            
            // Create node visual
            QGraphicsRectItem* nodeRect = graphScene_.addRect(x, y, nodeWidth, nodeHeight);
            nodeRect->setBrush(QBrush(Qt::white));
            nodeRect->setPen(QPen(Qt::black));
            
            // Add node name text
            QGraphicsTextItem* nameText = graphScene_.addText(QString::fromStdString(node->getName()));
            nameText->setPos(x + 5, y + 5);
            
            // Add operation type text
            QGraphicsTextItem* typeText = graphScene_.addText(
                QString::fromStdString(node->getOperation()->getType()));
            typeText->setPos(x + 5, y + 25);
        }
    }
    
    // Add edges between nodes
    for (const auto& node : graph_->getNodes()) {
        QPointF sourcePos = nodePositions[node->getName()];
        sourcePos.setX(sourcePos.x() + nodeWidth);
        sourcePos.setY(sourcePos.y() + nodeHeight / 2);
        
        for (const auto& outputNode : node->getOutputs()) {
            QPointF targetPos = nodePositions[outputNode->getName()];
            targetPos.setY(targetPos.y() + nodeHeight / 2);
            
            // Draw edge
            graphScene_.addLine(sourcePos.x(), sourcePos.y(), targetPos.x(), targetPos.y());
        }
    }
    

    // Adjust view to show all items
    ui->graphView->fitInView(graphScene_.itemsBoundingRect(), Qt::KeepAspectRatio);
    ui->graphView->setRenderHint(QPainter::Antialiasing);
}

void MainWindow::onImportModel() {
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Import Model"), "", tr("Model Files (*.pth *.pt *.pb *.onnx);;All Files (*)"));
    
    if (fileName.isEmpty()) {
        return;
    }
    
    spdlog::info("Importing model from file: {}", fileName.toStdString());
    
    try {
        // Here we would normally call into the OpenOptimizer API to import the model
        // For demonstration, we'll create a dummy graph
        auto graph = std::make_shared<ir::ComputationGraph>();
        setComputationGraph(graph);
        
        QMessageBox::information(this, tr("Import Successful"),
                                tr("Model has been imported successfully."));
    } catch (const std::exception& e) {
        spdlog::error("Error importing model: {}", e.what());
        QMessageBox::critical(this, tr("Import Error"),
                            tr("Failed to import model: %1").arg(e.what()));
    }
}

void MainWindow::onRunOptimization() {
    if (!graph_) {
        QMessageBox::warning(this, tr("No Model"),
                           tr("Please import a model first."));
        return;
    }
    
    spdlog::info("Running optimization");
    
    try {
        // Here we would normally call into the OpenOptimizer API to run optimizations
        // For demonstration, we'll just update the visualization
        updateVisualization();
        
        QMessageBox::information(this, tr("Optimization Complete"),
                                tr("Optimization completed successfully."));
    } catch (const std::exception& e) {
        spdlog::error("Error during optimization: {}", e.what());
        QMessageBox::critical(this, tr("Optimization Error"),
                            tr("Failed to optimize model: %1").arg(e.what()));
    }
}

void MainWindow::onGenerateCode() {
    if (!graph_) {
        QMessageBox::warning(this, tr("No Model"),
                           tr("Please import a model first."));
        return;
    }
    
    QString dir = QFileDialog::getExistingDirectory(this, tr("Select Output Directory"),
                                                    "", QFileDialog::ShowDirsOnly);
    
    if (dir.isEmpty()) {
        return;
    }
    
    // Get the target platform
    QStringList items;
    items << tr("CPU") << tr("GPU") << tr("Edge Device");
    bool ok;
    QString item = QInputDialog::getItem(this, tr("Select Target"),
                                         tr("Target Platform:"), items, 0, false, &ok);
    if (!ok || item.isEmpty()) {
        return;
    }
    
    codegen::TargetType target;
    if (item == "CPU") {
        target = codegen::TargetType::CPU;
    } else if (item == "GPU") {
        target = codegen::TargetType::GPU;
    } else {
        target = codegen::TargetType::Edge;
    }
    
    spdlog::info("Generating code for target {} at {}", static_cast<int>(target), dir.toStdString());
    
    try {
        // Here we would normally call into the OpenOptimizer API to generate code
        QMessageBox::information(this, tr("Code Generation Complete"),
                                tr("Code generation completed successfully."));
    } catch (const std::exception& e) {
        spdlog::error("Error during code generation: {}", e.what());
        QMessageBox::critical(this, tr("Code Generation Error"),
                            tr("Failed to generate code: %1").arg(e.what()));
    }
}

void MainWindow::onNodeSelected(const QString& nodeName) {
    // Display node details in the properties panel
    if (!graph_) {
        return;
    }
    
    auto node = graph_->getNode(nodeName.toStdString());
    if (!node) {
        return;
    }
    
    // Update properties view with node details
    ui->propertiesView->clear();
    
    QTreeWidgetItem* nameItem = new QTreeWidgetItem(ui->propertiesView);
    nameItem->setText(0, "Name");
    nameItem->setText(1, QString::fromStdString(node->getName()));
    
    QTreeWidgetItem* typeItem = new QTreeWidgetItem(ui->propertiesView);
    typeItem->setText(0, "Operation Type");
    typeItem->setText(1, QString::fromStdString(node->getOperation()->getType()));
    
    QTreeWidgetItem* inputsItem = new QTreeWidgetItem(ui->propertiesView);
    inputsItem->setText(0, "Inputs");
    inputsItem->setText(1, QString::number(node->getInputs().size()));
    
    QTreeWidgetItem* outputsItem = new QTreeWidgetItem(ui->propertiesView);
    outputsItem->setText(0, "Outputs");
    outputsItem->setText(1, QString::number(node->getOutputs().size()));
}

} // namespace visualization
} // namespace openoptimizer