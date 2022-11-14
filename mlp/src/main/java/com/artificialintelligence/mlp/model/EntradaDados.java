package com.artificialintelligence.mlp.model;

public class EntradaDados {
    private CalculationParameters calculationParameters;
    private String[][] trainingData;

    public EntradaDados() {
    }

    public EntradaDados(CalculationParameters calculationParameters, String[][] trainingData) {
        this.calculationParameters = calculationParameters;
        this.trainingData = trainingData;
    }

    public CalculationParameters getCalculationParameters() {
        return calculationParameters;
    }

    public String[][] getTrainingData() {
        return trainingData;
    }
}
