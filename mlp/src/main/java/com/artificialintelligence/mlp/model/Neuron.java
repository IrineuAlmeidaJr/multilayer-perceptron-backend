package com.artificialintelligence.mlp.model;

public class Neuron {
    private float net;
    private float i;
    private float error;
    private Neuron edges[];
    private float weights[];
    private int TL;

    public Neuron() {
    }

    public Neuron(Neuron[] edges, float[] weights, int TL) {
        this.net = 0;
        this.i = 0;
        this.error = 0;
        this.edges = edges;
        this.weights = weights;
        this.TL = TL;
    }
}
