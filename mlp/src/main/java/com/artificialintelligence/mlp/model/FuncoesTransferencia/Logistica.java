package com.artificialintelligence.mlp.model.FuncoesTransferencia;

public class Logistica {

    public double calcularFuncaoSaida(double net) {
        double E = Math.exp(1);
        return 1 / (1 + Math.pow(E, (-net)));
    }

    public double derivada(double saida) {
        return saida * (1 - saida);
    }

}
