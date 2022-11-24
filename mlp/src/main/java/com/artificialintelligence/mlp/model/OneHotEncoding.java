package com.artificialintelligence.mlp.model;

import java.util.Arrays;

public class OneHotEncoding {

    private boolean achouElemento(String classes[], String novaClasse) {
        // REFATORAR
        int tamanho = classes.length;
        int pos = 0;
        // Busca Exaustiva - Da para melhoras - REFATORAR
        while (pos < tamanho && !novaClasse.equalsIgnoreCase(classes[pos])) {
            pos++;
        }
        if (pos < tamanho) {
            return true;
        }
        return false;
    }

    private int buscaPos(String classes[], String novaClasse) {
        // Busca Exaustiva - Da para melhoras - REFATORAR
        int tamanho = classes.length;
        int pos = 0;
        while (pos < tamanho && !novaClasse.equalsIgnoreCase(classes[pos])) {
            pos++;
        }
        if (pos < tamanho) {
            return pos;
        }
        return -1;

    }

    public String[] retornarClasses(String dados[][], int numSaidas) {
        int numLinhas = dados.length;
        int numColunas = dados[0].length;
        int posClasse = numColunas-1;
        int TLC = 0;
        String[] classes = new String[numSaidas];
        String novaClasse;
        for (int i=1; i < numLinhas; i++) {
            novaClasse = dados[i][posClasse];
            if (!achouElemento(classes, novaClasse)) {
                classes[TLC++] = novaClasse;
            }
        }
        Arrays.sort(classes);

        return classes;
    }

    public double[][] tratarDados(String matrizDadosFront[][], int numSaidas, String[] classes,
                                  int funcaoTransferencia) {
        int numLinhas = matrizDadosFront.length;
        int numColunas = matrizDadosFront[0].length;
        int posClasse = numColunas-1;
        int qtdeNovasColunas = (numColunas + numSaidas) - 1;
        double [][]oneHotEnconding = new double[numLinhas-1][qtdeNovasColunas];

        // - Preencer as classes na Matriz One Hot Encode
        String tempClasse;
        int posAchouClasse;
        if (funcaoTransferencia != 3) {
            for (int i = 1, linha = 0; i < numLinhas; i++) {
                for (int j = 0; j < posClasse; j++) {
                    oneHotEnconding[linha][j] = Double.parseDouble(matrizDadosFront[i][j]);
                }
                tempClasse = matrizDadosFront[i][posClasse];
                posAchouClasse = buscaPos(classes, tempClasse);
                if (posAchouClasse != -1) {
                    // Insere na posição o valor 1 conforme One Hot Encode
                    oneHotEnconding[linha][posClasse + posAchouClasse] = 1;
                }
                linha++;
            }
        } else { // Muda Para Tangente Hiperbolica
            for (int i = 1, linha = 0; i < numLinhas; i++) {
                for (int j = 0; j < posClasse; j++) {
                    oneHotEnconding[linha][j] = Double.parseDouble(matrizDadosFront[i][j]);
                }
                for (int j = posClasse; j < qtdeNovasColunas; j++) {
                    oneHotEnconding[linha][j] = -1;
                }
                tempClasse = matrizDadosFront[i][posClasse];
                posAchouClasse = buscaPos(classes, tempClasse);
                if (posAchouClasse != -1) {
                    // Insere na posição o valor 1 conforme One Hot Encode
                    oneHotEnconding[linha][posClasse + posAchouClasse] = 1;
                }
                linha++;
            }
        }

        return oneHotEnconding;
    }


    public void Exibicao(double[][] dadosOneHotEncoding) {
        for (int i=0; i<dadosOneHotEncoding.length; i++) {
            for (int j=0; j<dadosOneHotEncoding[0].length; j++) {
                System.out.printf("%.4f   ", dadosOneHotEncoding[i][j]);
            }
            System.out.printf("\n");
        }
    }

}
