package com.artificialintelligence.mlp.model;

public class Neuronio {
    private double net;
    private double saida;
    private double erro;

    public Neuronio(double net, double saida, double erro) {
        this.net = net;
        this.saida = saida;
        this.erro = erro;
    }

    public double getNet() {
        return net;
    }

    public void setNet(double net) {
        this.net = net;
    }

    public double getSaida() {
        return saida;
    }

    public void setSaida(double saida) {
        this.saida = saida;
    }

    public double getErro() {
        return erro;
    }

    public void setErro(double erro) {
        this.erro = erro;
    }
}


