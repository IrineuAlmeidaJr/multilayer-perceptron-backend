package com.artificialintelligence.mlp.model.FuncoesTransferencia;

public class Linear {

    public double calcularFuncaoSaida(double net) {
        return net / 10;
    }

    public double derivada() {
        return 1.0/10.0;
    }

}
