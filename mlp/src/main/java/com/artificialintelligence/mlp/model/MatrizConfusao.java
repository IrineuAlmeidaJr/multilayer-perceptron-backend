package com.artificialintelligence.mlp.model;

public class MatrizConfusao {
    private int[][] matriz;
    private String[] classes;
    private double acuracia;

    public MatrizConfusao(int linhaColuna, String[] classes) {
        this.matriz = new int[linhaColuna][linhaColuna];
        this.classes = classes;
        this.acuracia = 0;
    }

    public int[][] getMatriz() {
        return matriz;
    }

    public void setMatriz(int linha, int coluna) {
        this.matriz[linha][coluna]++;
    }

    public String[] getClasses() {
        return classes;
    }

    public double getAcuracia() {
        return acuracia;
    }

    public void setAcuracia(double acuracia) {
        this.acuracia = acuracia;
    }
}
