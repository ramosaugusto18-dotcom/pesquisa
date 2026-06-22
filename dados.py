# ============================================================
# dados.py — Todas as pesquisas do modelo
# Para adicionar uma nova pesquisa: basta incluir as linhas
# no dicionário PESQUISAS com o formato abaixo.
# ============================================================

# Candidatos e cores
CANDIDATOS = ['Alberto Neto', 'Eduardo Braga', 'Plinio Valerio',
              'Wilson Lima', 'Marcos Rotta', 'Marcelo Ramos']

PARTIDOS = {
    'Alberto Neto':   'PL',
    'Eduardo Braga':  'MDB',
    'Plinio Valerio': 'PSDB',
    'Wilson Lima':    'UNIAO',
    'Marcos Rotta':   'Avante',
    'Marcelo Ramos':  'PT',
}

CORES = {
    'Alberto Neto':   '#f85149',
    'Eduardo Braga':  '#58a6ff',
    'Plinio Valerio': '#e3b341',
    'Wilson Lima':    '#3fb950',
    'Marcos Rotta':   '#8b949e',
    'Marcelo Ramos':  '#ff7b72',
}

# ============================================================
# PESQUISAS
# Formato de cada entrada:
#   'NomeInstituto': {
#       'tse':    'AM-XXXXX/2026',
#       'campo':  'DD/mmm',
#       'n':      1500,
#       'margem': 3.0,
#       'metodo': 'Presencial / Online / CATI / URA',
#       'tipo':   '1v'  (só 1º voto)
#                 'optB' (consolidado 1+2 normalizado para 100%)
#       'dados': {
#           'Alberto Neto': 28.0,
#           ...
#       }
#   }
# ============================================================

PESQUISAS = {

    'OpManauara': {
        'tse': 'AM-05275/2026',
        'campo': '01/mar',
        'n': 1200,
        'margem': 3.0,
        'metodo': 'Presencial',
        'tipo': '1v',
        'dados': {
            'Alberto Neto':   19.0,
            'Eduardo Braga':  23.0,
            'Plinio Valerio': 11.0,
            'Marcelo Ramos':   9.0,
            'Marcos Rotta':   12.0,
        }
    },

    'Quaest': {
        'tse': 'AM-01091/2026',
        'campo': '07/mar',
        'n': 1500,
        'margem': 3.0,
        'metodo': 'Presencial',
        'tipo': '1v',
        'dados': {
            'Alberto Neto':   28.0,
            'Eduardo Braga':  28.0,
            'Plinio Valerio': 14.0,
            'Marcelo Ramos':  10.0,
            'Marcos Rotta':    6.0,
        }
    },

    'AtlasIntel Mar': {
        'tse': 'AM-06921/2026',
        'campo': '10/mar',
        'n': 1138,
        'margem': 3.0,
        'metodo': 'Online RDR',
        'tipo': '1v',
        'dados': {
            'Alberto Neto':   24.0,
            'Eduardo Braga':  19.3,
            'Plinio Valerio': 17.4,
            'Marcelo Ramos':  15.3,
            'Marcos Rotta':    2.4,
        }
    },

    'RealTime': {
        'tse': 'AM-00383/2026',
        'campo': '13/mar',
        'n': 1500,
        'margem': 2.0,
        'metodo': 'Telefone',
        'tipo': '1v',
        'dados': {
            'Alberto Neto':   28.4,
            'Eduardo Braga':  17.0,
            'Plinio Valerio': 14.0,
            'Marcelo Ramos':   8.0,
            'Marcos Rotta':    9.0,
        }
    },

    'Veritá Mar': {
        'tse': 'AM-04742/2026',
        'campo': '18/mar',
        'n': 1220,
        'margem': 3.0,
        'metodo': 'URA',
        'tipo': '1v',
        'dados': {
            'Alberto Neto':   27.0,
            'Eduardo Braga':  20.7,
            'Plinio Valerio':  6.4,
            'Marcelo Ramos':   8.8,
            'Marcos Rotta':    4.3,
        }
    },

    'Iveritas': {
        'tse': 'AM-03018/2026',
        'campo': '20/mar',
        'n': 2604,
        'margem': 3.0,
        'metodo': 'Presencial',
        'tipo': 'optB',
        # 1º voto: Neto=36.15, Braga=28.09, Plinio=11.57
        # 2º voto: Neto=12.11, Braga=10.92, Plinio=23.02
        # Consolidado Opção B normalizado para 100%
        'dados': {
            'Alberto Neto':   31.78,
            'Eduardo Braga':  25.69,
            'Plinio Valerio': 22.78,
            'Marcelo Ramos':   9.88,
            'Marcos Rotta':    7.24,
        }
    },

    'AtlasIntel Mai': {
        'tse': 'AM-04/2026',
        'campo': '05/mai',
        'n': 1415,
        'margem': 3.0,
        'metodo': 'Online RDR',
        'tipo': 'optB',
        'dados': {
            'Alberto Neto':   23.7,
            'Eduardo Braga':  21.6,
            'Plinio Valerio': 15.5,
            'Marcelo Ramos':  17.9,
            'Marcos Rotta':    4.7,
            'Wilson Lima':     7.5,
        }
    },

    'Census': {
        'tse': 'AM-02868/2026',
        'campo': '10/mai',
        'n': 2000,
        'margem': 2.2,
        'metodo': 'CATI (Telefone)',
        'tipo': 'optB',
        # 1º: Braga=33, Neto=27, Plinio=13, Wilson=13, Rotta=6, Ramos=8
        # 2º: Braga=16, Neto=15, Plinio=16, Wilson=12, Rotta=13, Ramos=11
        'dados': {
            'Alberto Neto':   22.95,
            'Eduardo Braga':  26.78,
            'Plinio Valerio': 15.85,
            'Wilson Lima':    13.66,
            'Marcos Rotta':   10.38,
            'Marcelo Ramos':  10.38,
        }
    },

    'IPEN': {
        'tse': 'AM-07612/2026',
        'campo': '22/mai',
        'n': 1200,
        'margem': 2.8,
        'metodo': 'Presencial',
        'tipo': 'optB',
        # 1º: Braga=37.4, Neto=17.8, Wilson=12.6, Plinio=6.9, Rotta=5.5, Ramos=4.3
        # 2º: Braga=13.0, Neto=12.1, Wilson=11.7, Plinio=8.5, Rotta=8.9, Ramos=8.8
        'dados': {
            'Eduardo Braga':  34.17,
            'Alberto Neto':   20.27,
            'Wilson Lima':    16.47,
            'Plinio Valerio': 10.44,
            'Marcos Rotta':    9.76,
            'Marcelo Ramos':   8.88,
        }
    },

    'Veritá Mai': {
        'tse': 'AM-04742/2026 (2ª rodada)',
        'campo': '28/mai',
        'n': 1220,
        'margem': 2.8,
        'metodo': 'Presencial',
        'tipo': 'optB',
        # 1º: Neto=38.9, Braga=27.5, Ramos=12.8, Plinio=10.1, Wilson=6.2, Rotta=2.0
        # 2º: Braga=22.4, Plinio=20.7, Ramos=17.5, Neto=17.4, Wilson=8.4, Rotta=5.3
        'dados': {
            'Alberto Neto':   29.76,
            'Eduardo Braga':  26.37,
            'Plinio Valerio': 16.28,
            'Marcelo Ramos':  16.01,
            'Wilson Lima':     7.72,
            'Marcos Rotta':    3.86,
        }
    },

    # ============================================================
    # ADICIONAR NOVAS PESQUISAS AQUI — copie o bloco acima
    # e preencha com os novos dados. O modelo atualiza sozinho.
    # ============================================================

}

# Rejeição (última pesquisa disponível — Veritá Mai)
REJEICAO = {
    'Alberto Neto':   36.6,
    'Eduardo Braga':  23.6,
    'Marcelo Ramos':  16.0,
    'Wilson Lima':    12.5,
    'Plinio Valerio':  1.3,
    'Marcos Rotta':    0.6,
}
