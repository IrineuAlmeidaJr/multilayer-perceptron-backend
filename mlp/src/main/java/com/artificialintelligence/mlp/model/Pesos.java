package com.artificialintelligence.mlp.model;

import java.util.ArrayList;

public class Pesos {
    private ArrayList<Integer> posEntrada;
    private ArrayList<Integer> posSaida;
    private ArrayList<Double> peso;

    public Pesos() {
        this.posEntrada = new ArrayList<Integer>();
        this.posSaida = new ArrayList<Integer>();
        this.peso = new ArrayList<Double>();
    }

    public int getPosEntrada(int pos) {
        return posEntrada.get(pos);
    }

    public ArrayList<Integer> getAllPosEntrada() {
        return posEntrada;
    }

    public void setPosEntrada(int posEntrada) {
        this.posEntrada.add(posEntrada);
    }

    public int getPosSaida(int pos) {
        return posSaida.get(pos);
    }

    public ArrayList<Integer> getAllPosSaida() {
        return posSaida;
    }

    public void setPosSaida(int posSaida) {
        this.posSaida.add(posSaida);
    }

    public double getPeso(int pos) {
        return this.peso.get(pos);
    }

    public ArrayList<Double> getAllPeso() {
        return peso;
    }

    public void setPeso(double peso) {
        this.peso.add(peso);
    }

    public void inicializarPesos(int totalEntradas, int totalSaidas, int numCamadaOculta) {
        // ---> Criar veto PESOS
        int posSaida = totalEntradas * numCamadaOculta + totalEntradas;
        int qtdePesosEntrada = totalEntradas * totalEntradas;
        int qtdePesosSaida = totalSaidas * totalEntradas;
        int qtdeTotalPesos;
        if (numCamadaOculta == 1) {
            int posInserir = 0;
            qtdeTotalPesos = qtdePesosEntrada + qtdePesosSaida;

            // Para Entrada
            for (int entrada=0; entrada < totalEntradas; entrada++) {
                for (int k=0; k < totalEntradas; k++) {
                    setPosEntrada(entrada);
                    setPosSaida(k+totalEntradas);
                    setPeso(((Math.random() * 2) - 1));
                }
            }

            // Pesos SAIDA
            int posUltimaCamadaOculta = posSaida - totalEntradas;
            for (int entrada=posUltimaCamadaOculta; entrada < posSaida; entrada++) {
                for (int k=0; k < totalSaidas; k++) {
                    setPosEntrada(entrada);
                    setPosSaida(k+posSaida);
                    setPeso(((Math.random() * 2) - 1));
                }
            }

        } else {
            int posInserir = 0;
            int qtdePesosIntermediarios = qtdePesosEntrada * (numCamadaOculta-1);
            qtdeTotalPesos = qtdePesosEntrada +
                    qtdePesosIntermediarios +
                    qtdePesosSaida;


            qtdeTotalPesos = qtdePesosEntrada + qtdePesosSaida;
            int saida;

            // Pesos ENTRADA
            for (int entrada=0; entrada < totalEntradas; entrada++) {
                for (int k=0; k < totalEntradas; k++) {
                    setPosEntrada(entrada);
                    setPosSaida(k+totalEntradas);
                    setPeso(((Math.random() * 2) - 1));
                }
            }

            // Pesos INTERMEDIARIO
            int posInicioOculta;
            int posSaidaOculta;
            int posLimiteOculta = totalEntradas + totalEntradas;
            for (int camadaOculta = 0; camadaOculta < numCamadaOculta-1; camadaOculta++) {
                posSaidaOculta =  posLimiteOculta + (totalEntradas * camadaOculta);
                posInicioOculta = posSaidaOculta - totalEntradas;
                for (int entrada = posInicioOculta; entrada < posSaidaOculta; entrada++) {
                    for (int k = 0; k < totalEntradas; k++) {
                        setPosEntrada(entrada);
                        setPosSaida(k + posSaidaOculta);
                        setPeso(((Math.random() * 2) - 1));
                    }
                }
            }

            // Pesos SAIDA
            int posUltimaCamadaOculta = posSaida - totalEntradas;
            for (int entrada=posUltimaCamadaOculta; entrada < posSaida; entrada++) {
                for (int k=0; k < totalSaidas; k++) {
                    setPosEntrada(entrada);
                    setPosSaida(k+posSaida);
                    setPeso(((Math.random() * 2) - 1));
                }
            }
        }
    }
}