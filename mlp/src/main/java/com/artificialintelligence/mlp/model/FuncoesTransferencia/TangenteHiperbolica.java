package com.artificialintelligence.mlp.model.FuncoesTransferencia;

public class TangenteHiperbolica {

    public double calcularFuncaoSaida(double net) {
        return Math.tanh(net);
    }

    public double derivada(double saida) {
        return 1 - (Math.pow(saida, 2));
    }




}
