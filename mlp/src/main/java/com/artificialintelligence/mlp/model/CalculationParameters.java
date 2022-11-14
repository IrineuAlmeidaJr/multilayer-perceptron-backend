package com.artificialintelligence.mlp.model;

public class CalculationParameters {
    private int inputLayer;
    private int outputLayer;
    private int hiddenLayer;
    private float errorValue;
    private int numberIterations;
    private float learningRate;
    private int transferFunction;

    public CalculationParameters() {
    }

    public CalculationParameters(int inputLayer, int outputLayer, int hiddenLayer, float errorValue, int numberIterations, float learningRate, int transferFunction) {
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.hiddenLayer = hiddenLayer;
        this.errorValue = errorValue;
        this.numberIterations = numberIterations;
        this.learningRate = learningRate;
        this.transferFunction = transferFunction;
    }

    public int getInputLayer() {
        return inputLayer;
    }

    public int getOutputLayer() {
        return outputLayer;
    }

    public int getHiddenLayer() {
        return hiddenLayer;
    }

    public float getErrorValue() {
        return errorValue;
    }

    public int getNumberIterations() {
        return numberIterations;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public int getTransferFunction() {
        return transferFunction;
    }
}
