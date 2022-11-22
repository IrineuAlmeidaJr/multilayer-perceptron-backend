package com.artificialintelligence.mlp.model;

public class NormalizacaoValores {

    private double min;
    private double max;

    public NormalizacaoValores(double min, double max) {
        this.min = min;
        this.max = max;
    }

    public double getMin() {
        return min;
    }

    public double getMax() {
        return max;
    }

    public void setMin(double min) {
        this.min = min;
    }

    public void setMax(double max) {
        this.max = max;
    }

}
