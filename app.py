import streamlit as st
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import re
from itertools import combinations
from collections import Counter, defaultdict

st.set_page_config(page_title="Club Chelero", layout="wide")

pio.renderers.default = 'iframe'

st.sidebar.header("Carga de Archivos")
archivo_historicos = st.sidebar.file_uploader("Cargar archivo de histórico de puntos", type=["xlsx"])
archivo_skus = st.sidebar.file_uploader("Cargar archivo de compras por SKU", type=["xlsx"])
archivo_usuarios = st.sidebar.file_uploader("Cargar archivo de datos de usuarios", type=["xlsx"])
archivo_utilidad = st.sidebar.file_uploader("Cargar archivo de margen por producto", type=["xlsx"])
archivo_tiendas = st.sidebar.file_uploader("Cargar archivo de precios por tienda", type=["xlsx"])

@st.cache_data
def load_data_manual(archivo_historicos, archivo_skus, archivo_usuarios, archivo_utilidad, archivo_tiendas):
    if archivo_historicos and archivo_skus and archivo_usuarios and archivo_utilidad and archivo_tiendas:
        df_historicos = pd.read_excel(archivo_historicos)
        df_skus = pd.read_excel(archivo_skus)
        df_usuarios = pd.read_excel(archivo_usuarios)
        df_utilidad = pd.read_excel(archivo_utilidad)
        df_tiendas = pd.read_excel(archivo_tiendas)
        return df_historicos, df_skus, df_usuarios, df_utilidad, df_tiendas
    else:
        return None, None, None, None, None

df_historicos, df_skus, df_usuarios, df_utilidad, df_tiendas = load_data_manual(
    archivo_historicos, archivo_skus, archivo_usuarios, archivo_utilidad, archivo_tiendas)

if all(df is not None for df in [df_historicos, df_skus, df_usuarios, df_utilidad, df_tiendas]):

    df_historicos.rename(columns={'Nº cuenta': 'id_cliente'}, inplace=True)
    df_skus.rename(columns={'No. Cuenta Miembro': 'id_cliente'}, inplace=True)
    df_usuarios.rename(columns={'tel_usuario': 'id_cliente'}, inplace=True)

    dicc_grupo_etario = df_usuarios.groupby('Grupo Etario')['id_cliente'].apply(list).to_dict()
    dicc_id_to_grupo = {
        id_usuario: grupo for grupo, ids in dicc_grupo_etario.items() for id_usuario in ids
    }

    df_utilidad = df_utilidad.rename(columns={'ID': 'SKU'})
    df_utilidad['%'] = df_utilidad.apply(
        lambda row: row['Utilidad'] / row['Venta'] if row['Venta'] != 0 else 0,
        axis=1
    )
    dicc_sku_info = df_utilidad.set_index('SKU')[[
        'Categoria', 'Sub Categoria', 'Familia', 'Descripción', '%'
    ]].to_dict(orient='index')

    dict_tiendas_precio = dict(zip(df_tiendas['Store'], df_tiendas['Price Group Code']))

    def construir_df_base(df_historicos, df_skus, df_utilidad):
        df_h = df_historicos.copy()
        df_s = df_skus.copy()

        df_s['Cantidad'] = df_s['Cantidad'].abs()
        df_s['Valor Bruto'] = df_s['Valor Bruto'].abs()

        df_h['Fecha'] = pd.to_datetime(df_h['Fecha'])
        df_s['Fecha'] = pd.to_datetime(df_s['Fecha'])

        df_h['clave'] = (
            df_h['Fecha'].astype(str) + "_" +
            df_h['No. Tienda'].astype(str) + "_" +
            df_h['Nº Transacción'].astype(str)
        )

        df_s['clave'] = (
            df_s['Fecha'].astype(str) + "_" +
            df_s['No. Tienda'].astype(str) + "_" +
            df_s['Nº Transacción'].astype(str)
        )

        columnas_historico = df_h.drop(columns=['No. Contacto', 'No. Tarjeta'])

        columnas_skus = df_s[[
            'clave', 'No. Producto', 'Descripción', 'Cantidad',
            'Valor Bruto', 'Importe Descuento',
            'Código Categoría Producto', 'Cód. grupo producto'
        ]]

        df_base = columnas_historico.merge(columnas_skus, on='clave', how='inner')
        df_base = df_base.rename(columns={'No. Producto': 'SKU'})

        df_base['Grupo Etario'] = df_base['id_cliente'].map(dicc_id_to_grupo)
        df_base['Categoria'] = df_base['SKU'].map(lambda x: dicc_sku_info.get(x, {}).get('Categoria'))
        df_base['Sub Categoria'] = df_base['SKU'].map(lambda x: dicc_sku_info.get(x, {}).get('Sub Categoria'))
        df_base['Familia'] = df_base['SKU'].map(lambda x: dicc_sku_info.get(x, {}).get('Familia'))
        df_base['Descripción Detallada'] = df_base['SKU'].map(lambda x: dicc_sku_info.get(x, {}).get('Descripción'))
        df_base['% Utilidad'] = df_base['SKU'].map(lambda x: dicc_sku_info.get(x, {}).get('%'))
        df_base['Grupo Precio'] = df_base['No. Tienda'].map(dict_tiendas_precio)
        df_base['Grupo Precio'] = df_base['Grupo Precio'].fillna('DESCONOCIDO')

        return df_base

    df_base = construir_df_base(df_historicos, df_skus, df_utilidad)

    def calcular_kpis(df):
        
        total_ventas = df['Valor Bruto'].sum()
        clientes_unicos = df['id_cliente'].nunique()
        productos_vendidos = df['Cantidad'].sum()
        return total_ventas, clientes_unicos, productos_vendidos

    def mostrar_kpis(df_base):
        
        total_ventas, clientes_unicos, productos_vendidos = calcular_kpis(df_base)
    
        # --- Periodo de análisis ---
        fecha_min = pd.to_datetime(df_base['Fecha']).min().date()
        fecha_max = pd.to_datetime(df_base['Fecha']).max().date()
        periodo_analizado = f"{fecha_min} a {fecha_max}"
    
        # --- Visualización en Streamlit ---
        col_titulo, col_periodo = st.columns([2, 1])
        with col_titulo:
            st.subheader("📈 Visión General")
        with col_periodo:
            st.markdown(f"**🗓️ Periodo Analizado:**  \n{periodo_analizado}")
    
        col1, col2, col3 = st.columns(3)
        col1.metric("💵 Total Ventas", f"${total_ventas:,.0f}")
        col2.metric("👥 Clientes Únicos", f"{clientes_unicos:,}")
        col3.metric("📦 Productos Vendidos", f"{int(productos_vendidos):,}")

    mostrar_kpis(df_base)

    
    def mostrar_comparativa_ventas(df_base):
        st.subheader("Ventas por Tienda")
    
        top_n = st.selectbox("¿Cuántas tiendas deseas visualizar?", options=[5, 10, 15, 20, 30, 50], index=1)
    
        resumen_tienda = (
            df_base.groupby('No. Tienda')
            .agg(Total_Ventas=('Valor Bruto', 'sum'))
            .reset_index()
            .sort_values('Total_Ventas', ascending=False)
            .head(top_n)
        )
    
        fig = px.bar(
            resumen_tienda,
            x='No. Tienda',
            y='Total_Ventas',
            title=f"Top {top_n} Tiendas por Ventas",
            labels={'No. Tienda': 'Tienda', 'Total_Ventas': 'Ventas ($)'},
            text_auto='.2s'
        )
        fig.update_layout(xaxis_title="", yaxis_title="Ventas ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    mostrar_comparativa_ventas(df_base)
    
    def mostrar_ventas_por_grupo_precio(df_base):
        st.subheader("Ventas por Grupo Precio")
    
        resumen_precio = (
            df_base.groupby('Grupo Precio')
            .agg(Ventas_Totales=('Valor Bruto', 'sum'))
            .reset_index()
            .sort_values('Ventas_Totales', ascending=False)
        )
    
        fig = px.bar(
            resumen_precio,
            x='Grupo Precio',
            y='Ventas_Totales',
            title='Comparativa de Ventas por Grupo Precio',
            labels={'Grupo Precio': 'Grupo Precio', 'Ventas_Totales': 'Ventas ($)'},
            text_auto='.2s'
        )
        fig.update_layout(xaxis_title="", yaxis_title="Ventas ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    mostrar_ventas_por_grupo_precio(df_base)
    
    def resumen_mensual_ventas_transacciones(df_base):
            # Asegurar que la columna Fecha esté en formato datetime
            df_base = df_base.copy()
            df_base['Fecha'] = pd.to_datetime(df_base['Fecha'])
        
            # Crear columnas de año y mes
            df_base['Año'] = df_base['Fecha'].dt.year
            df_base['Mes'] = df_base['Fecha'].dt.month
        
            # Agrupar por año y mes
            resumen = (
                df_base.groupby(['Año', 'Mes'])
                .agg(
                    Ventas_Totales=('Valor Bruto', 'sum'),
                    Transacciones_Totales=('Nº Transacción', pd.Series.nunique)
                )
                .reset_index()
            )
        
            promedio_ventas = resumen['Ventas_Totales'].mean()
            promedio_transacciones = resumen['Transacciones_Totales'].mean()
        
            return resumen, promedio_ventas, promedio_transacciones    
    
    def ticket_promedio_mensual(df_base):
            # Asegurarse de que la columna de fecha sea datetime
            df_base['Fecha'] = pd.to_datetime(df_base['Fecha'])
        
            # Consolidar el valor del ticket por transacción (suma del valor bruto)
            tickets = df_base.groupby(['Nº Transacción', 'Fecha'], as_index=False)['Valor Bruto'].sum()
        
            tickets['Año'] = tickets['Fecha'].dt.year
            tickets['Mes'] = tickets['Fecha'].dt.month
        
            #promedio por mes
            tickets_promedio = (
                tickets.groupby(['Año', 'Mes'])['Valor Bruto']
                .mean()
                .reset_index(name='Ticket Promedio')
            )
        
            return tickets_promedio
    
    def mostrar_evolucion_mensual(df_base):
            st.subheader("Evolución Mensual")
        
            # 1. Ventas y Transacciones Totales
            resumen, promedio_ventas, promedio_transacciones = resumen_mensual_ventas_transacciones(df_base)
        
            resumen['Periodo'] = resumen['Año'].astype(str) + "-" + resumen['Mes'].astype(str).str.zfill(2)
        
            col1, col2 = st.columns(2)
            col1.metric("Promedio Ventas Mensuales", f"${promedio_ventas:,.0f}")
            col2.metric("Promedio Transacciones Mensuales", f"{promedio_transacciones:,.0f}")
        
            # Gráfica de ventas con etiquetas
            # Gráfica de ventas con etiquetas y formato
            fig_ventas = px.line(
                resumen,
                x='Periodo',
                y='Ventas_Totales',
                title='📈 Evolución Mensual de Ventas',
                markers=True,
                labels={'Ventas_Totales': 'Ventas Totales'},
                text='Ventas_Totales'
            )
            fig_ventas.update_traces(textposition='top center', texttemplate='$%{y:,.1s}')
            fig_ventas.update_yaxes(tickformat='$,.1s')  # Formato tipo $1.2M
            st.plotly_chart(fig_ventas, use_container_width=True)
            
            # Gráfica de transacciones con etiquetas y formato
            fig_transacciones = px.line(
                resumen,
                x='Periodo',
                y='Transacciones_Totales',
                title='🔁 Evolución Mensual de Transacciones',
                markers=True,
                labels={'Transacciones_Totales': 'Transacciones'},
                text='Transacciones_Totales'
            )
            fig_transacciones.update_traces(textposition='top center', texttemplate='%{y:,}')
            fig_transacciones.update_yaxes(tickformat=',')  # Separador de miles
            st.plotly_chart(fig_transacciones, use_container_width=True)
            
            # Gráfica de ticket promedio con etiquetas y formato
            tickets_promedio = ticket_promedio_mensual(df_base)
            tickets_promedio['Periodo'] = tickets_promedio['Año'].astype(str) + "-" + tickets_promedio['Mes'].astype(str).str.zfill(2)
            
            fig_ticket = px.line(
                tickets_promedio,
                x='Periodo',
                y='Ticket Promedio',
                title='🧾 Evolución del Ticket Promedio Mensual',
                markers=True,
                text='Ticket Promedio'
            )
            fig_ticket.update_traces(textposition='top center', texttemplate='$%{y:,.0f}')
            fig_ticket.update_yaxes(tickformat='$,.0f')  # Formato monetario estándar
            st.plotly_chart(fig_ticket, use_container_width=True)
    
    
    mostrar_evolucion_mensual(df_base)
    
    def mostrar_productos_mas_vendidos(df_base):
            st.subheader("Productos Más Vendidos")
        
            # Agrupar por producto
            resumen_prod = (
                df_base.groupby(['SKU', 'Descripción'], as_index=False)
                .agg(
                    Ventas_Totales=('Valor Bruto', 'sum'),
                    Unidades_Totales=('Cantidad', 'sum')
                )
                .sort_values(by='Ventas_Totales', ascending=False)
            )
        
            # Selector: top N
            top_n = st.selectbox("Selecciona cuántos productos mostrar:", [5, 10, 15, 20], index=1)
        
            # Selector: tipo de métrica
            tipo_metrica = st.radio("Ordenar por:", ['Ventas Totales', 'Unidades Totales'], horizontal=True)
        
            if tipo_metrica == 'Ventas Totales':
                resumen_top = resumen_prod.nlargest(top_n, 'Ventas_Totales')
                y_col = 'Ventas_Totales'
                title = f"💰 Top {top_n} Productos por Ventas"
            else:
                resumen_top = resumen_prod.nlargest(top_n, 'Unidades_Totales')
                y_col = 'Unidades_Totales'
                title = f"📦 Top {top_n} Productos por Unidades Vendidas"
        
            fig = px.bar(
                resumen_top,
                x='Descripción',
                y=y_col,
                text_auto='.2s',
                title=title,
                labels={y_col: tipo_metrica},
            )
            fig.update_layout(xaxis_tickangle=-45, xaxis_title="", yaxis_title=tipo_metrica)
            st.plotly_chart(fig, use_container_width=True)
        
            # Mostrar tabla completa opcionalmente
            with st.expander("🔍 Ver tabla completa"):
                st.dataframe(resumen_prod, use_container_width=True)
        
    mostrar_productos_mas_vendidos(df_base)
    
    def analizar_segmentos_por_edad(df_base):
            resumen = df_base.groupby('Grupo Etario').agg(
                #Total_Clientes=('id_cliente', 'nunique'),
                Total_Transacciones=('Nº Transacción', 'nunique'),
                Total_Puntos=('Puntos', 'sum'),
                Total_Cantidad_Productos=('Cantidad', 'sum'),
                Valor_Total_Compras=('Valor Bruto', 'sum')
            ).reset_index().sort_values(by='Valor_Total_Compras', ascending=True)
            
            return resumen
    
    def productos_top_por_segmento(df_base, top_n=5):
            productos_top = (
                df_base.groupby(['Grupo Etario', 'Descripción'])
                .agg(Ventas_Totales=('Valor Bruto', 'sum'))
                .reset_index()
            )
            
            top_productos_segmento = productos_top.sort_values(['Grupo Etario','Ventas_Totales'], ascending=[True, False])
            
            # Obtener los top_n productos por segmento
            top_productos_segmento = top_productos_segmento.groupby('Grupo Etario').head(top_n)
            
            return top_productos_segmento
    
    def mostrar_analisis_por_segmento(df_base):
            st.header("Análisis por Segmento de Edad")
        
            # 1. Resumen general por grupo etario
            st.subheader("📊 Resumen General por Grupo Etario")
            segmentos = analizar_segmentos_por_edad(df_base)
            st.dataframe(segmentos, use_container_width=True)
        
            fig = px.bar(
                segmentos,
                x='Grupo Etario',
                y='Valor_Total_Compras',
                title='💵 Valor Total de Compras por Grupo Etario',
                labels={'Valor_Total_Compras': 'Valor Total ($)'},
                text_auto='.2s'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    mostrar_analisis_por_segmento(df_base)
    
    def mostrar_top_productos_por_segmento(df_base):
            st.subheader("Top Productos por Grupo Etario (Selección Individual)")
        
            grupos = df_base['Grupo Etario'].dropna().unique()
            grupos_ordenados = sorted(grupos)
        
            grupo_seleccionado = st.selectbox("Selecciona un grupo etario:", grupos_ordenados)
        
            top_n = st.slider("¿Cuántos productos mostrar?", min_value=3, max_value=15, value=5)
        
            top_productos_sg = productos_top_por_segmento(df_base, top_n=top_n)
        
            productos_filtrados = top_productos_sg[top_productos_sg['Grupo Etario'] == grupo_seleccionado]
        
            if not productos_filtrados.empty:
                fig = px.bar(
                    productos_filtrados,
                    x='Descripción',
                    y='Ventas_Totales',
                    title=f"🥇 Top {top_n} Productos - {grupo_seleccionado}",
                    labels={'Ventas_Totales': 'Ventas Totales ($)'},
                    text_auto='.2s'
                )
                fig.update_layout(xaxis_tickangle=-45, xaxis_title="", yaxis_title="Ventas Totales")
                st.plotly_chart(fig, use_container_width=True)
        
                with st.expander("🔍 Ver todos los productos de este grupo etario"):
                    todos_los_productos = (
                        df_base[df_base['Grupo Etario'] == grupo_seleccionado]
                        .groupby('Descripción', as_index=False)
                        .agg(Ventas_Totales=('Valor Bruto', 'sum'))
                        .sort_values(by='Ventas_Totales', ascending=False)
                    )
                    st.dataframe(todos_los_productos, use_container_width=True)
            else:
                st.warning("No hay datos disponibles para este grupo etario.")
    
    mostrar_top_productos_por_segmento(df_base)
    
    def detectar_cross_selling_detallado_flexible_streamlit(df_base):
        st.subheader("🔁 Análisis de Venta en Conjunto (Cross Selling)")
    
        # Obtener opciones únicas
        grupo_etario_opciones = ['Todos'] + sorted(df_base['Grupo Etario'].dropna().unique())
        categoria_opciones = ['Todas'] + sorted(df_base['Categoria'].dropna().unique())
        fecha_min = pd.to_datetime(df_base['Fecha'].min())
        fecha_max = pd.to_datetime(df_base['Fecha'].max())
    
        # --- Filtros en el cuerpo principal ---
        col1, col2, col3 = st.columns(3)
        with col1:
            grupo_etario = st.selectbox("Grupo Etario", grupo_etario_opciones)
        with col2:
            categoria_focal = st.selectbox("Categoría focal", categoria_opciones)
        with col3:
            min_apariciones = st.slider("Frecuencia mínima", 2, 100, 10)
    
        rango_fechas = st.date_input("Rango de fechas a analizar", [fecha_min, fecha_max])
    
        ejecutar = st.button("🔎 Ejecutar análisis")
    
        if ejecutar:
            st.info("Procesando combinaciones...")
    
            # Filtro
            df = df_base.copy()
            if grupo_etario != 'Todos':
                df = df[df['Grupo Etario'] == grupo_etario]
            if categoria_focal != 'Todas':
                categoria_focal = categoria_focal.upper()
            if rango_fechas:
                f1, f2 = pd.to_datetime(rango_fechas[0]), pd.to_datetime(rango_fechas[1])
                df = df[(df['Fecha'] >= f1) & (df['Fecha'] <= f2)]
    
            df = df[df['% Utilidad'] > 0]
    
            # Agrupar por ticket
            productos_por_ticket = df.groupby('clave')[['SKU', 'Categoria', 'Descripción', '% Utilidad']].apply(
                lambda x: x.drop_duplicates().to_dict('records')
            ).reset_index(name='Productos')
    
            combinaciones = []
            for _, row in productos_por_ticket.iterrows():
                productos = row['Productos']
    
                if categoria_focal != 'Todas':
                    focales = [p for p in productos if p['Categoria'] == categoria_focal]
                    otros = [p for p in productos if p['Categoria'] != categoria_focal]
                    pares = [(f, o) for f in focales for o in otros]
                else:
                    pares = list(combinations(productos, 2))
    
                for p1, p2 in pares:
                    key = tuple(sorted([p1['SKU'], p2['SKU']]))
                    combinaciones.append((key, p1, p2))
    
            conteo = Counter([c[0] for c in combinaciones])
    
            combinaciones_filtradas = [
                (sku_pair, p1, p2, freq)
                for (sku_pair, p1, p2), freq in zip(combinaciones, [conteo[c[0]] for c in combinaciones])
                if freq >= min_apariciones
            ]
    
            registros = []
            ya_vistos = set()
            for (sku1, sku2), p1, p2, freq in combinaciones_filtradas:
                key = tuple(sorted([sku1, sku2]))
                if key in ya_vistos:
                    continue
                ya_vistos.add(key)
    
                registros.append({
                    'SKU 1': p1['SKU'],
                    'Categoría 1': p1['Categoria'],
                    'Descripción 1': p1['Descripción'],
                    'Utilidad 1': p1['% Utilidad'],
                    'SKU 2': p2['SKU'],
                    'Categoría 2': p2['Categoria'],
                    'Descripción 2': p2['Descripción'],
                    'Utilidad 2': p2['% Utilidad'],
                    'Frecuencia': freq,
                    'Utilidad Promedio Combo': (p1['% Utilidad'] + p2['% Utilidad']) / 2
                })
    
            if not registros:
                st.warning("Combinaciones no encontradas")
            else:
                df_resultado = pd.DataFrame(registros)
                df_resultado = df_resultado.sort_values(by='Frecuencia', ascending=False).reset_index(drop=True)
                st.success(f"{len(df_resultado)} combinaciones detectadas.")
                st.dataframe(df_resultado)

    detectar_cross_selling_detallado_flexible_streamlit(df_base)

else:
    st.warning("Por favor carga los cinco archivos requeridos para visualizar el dashboard.")
