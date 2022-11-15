package com.artificialintelligence.mlp.model.FuncoesTransferencia;

public class Logistica {

    public double calcularFuncaoSaida(double net) {
        return 1 / (1 + Math.pow(Math.exp(1), (-net)));
    }

    public double derivada(double saida) {
        return saida * (1 - saida);
    }

}
