package com.artificialintelligence.mlp.model.FuncoesTransferencia;

public class Logistica {

    public double calcularFuncaoSaida(double net) {
        return 1.0 / (1.0 + Math.exp(-net));
    }

    public double derivada(double saida) {
        return saida * (1.0 - saida);
    }

}
